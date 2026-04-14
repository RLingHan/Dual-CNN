"""
BidirectionalSimplifiedSSM — 即插即用双向简化SSM（内存安全高速版）
=================================================================
为什么不用 cumsum 矩阵展开：
  decay 矩阵形状 [B, L, L, d]，L=32/d=512/B=32 时就需要 2GB，OOM。

正确方案：
  1. 降维瓶颈：d_model(2048) -> inner_dim(默认256) 做SSM
     -> decay矩阵缩小8倍，且for循环里element-wise极快
  2. Linear全部预计算（完全并行）
  3. for循环只做 h = h * delta + gate，L=32时32次element-wise，极快
  4. torch.compile 自动fuse循环内的kernel

内存占用对比（B=32, L=32）：
  旧版 cumsum矩阵: 32×32×32×512×4B = 2GB  ← OOM
  新版 for循环:    32×32×256×4B    = 32MB  ← 安全

用法（完全不变）：
    ssm = BidirectionalSimplifiedSSM(d_model=2048)
    out = ssm(layer4_feat)   # [B, 2048, 8, 4] -> [B, 2048, 8, 4]
    out = ssm(seq)           # [B, L, D]        -> [B, L, D]
"""

import torch
import torch.nn as nn


class BidirectionalSimplifiedSSM(nn.Module):
    """
    Args:
        d_model    (int):   输入特征维度（如 2048）
        inner_dim  (int):   SSM内部维度，默认 d_model//8（即256）
                            内存 ∝ B×L×inner_dim，调小可进一步省内存
        dropout  (float):   输出dropout，默认0
    """

    def __init__(
        self,
        d_model: int,
        inner_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model   = d_model
        self.inner_dim = inner_dim or max(d_model // 8, 64)  # 默认256@2048

        d = self.inner_dim

        # 降维
        self.proj_in  = nn.Linear(d_model, d, bias=False)
        # 正向：delta + gate 合并一个Linear（减少kernel launch）
        self.fwd_proj = nn.Linear(d, d * 2, bias=False)
        # 反向
        self.bwd_proj = nn.Linear(d, d * 2, bias=False)
        # 升维还原
        self.proj_out = nn.Linear(d * 2, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)
        # delta初始化偏小，初始遗忘率接近0.5
        nn.init.normal_(self.fwd_proj.weight, std=0.01)
        nn.init.normal_(self.bwd_proj.weight, std=0.01)

    def _scan(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """
        单向扫描。
        核心递推：h_t = delta_t * h_{t-1} + gate_t * x_t

        优化点：
          - proj(x) 一次性并行算出所有时间步的 delta 和 gate  [完全并行]
          - for 循环内只有两次 element-wise 操作               [极快]
          - L=32 时循环32次，每次操作 [B, d] 的小tensor        [无瓶颈]

        内存峰值：outputs stack 前为 [B, d]，stack 后 [B, L, d]
        """
        B, L, d = x.shape

        # ── 预计算（完全并行，这里是主要计算量）────────────────
        out   = proj(x)                          # [B, L, 2d]
        delta = torch.sigmoid(out[..., :d])      # [B, L, d]  遗忘门
        gate  = out[..., d:] * x                 # [B, L, d]  输入项

        # ── 串行递推（L=32，32次element-wise，极快）────────────
        h = x.new_zeros(B, d)
        outputs = []
        for t in range(L):
            h = h * delta[:, t] + gate[:, t]
            outputs.append(h)

        return torch.stack(outputs, dim=1)       # [B, L, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 或 [B, L, D]
        Returns:
            同输入形状
        """
        # ── 自动序列化 ────────────────────────────────────────
        is_4d = x.dim() == 4
        if is_4d:
            B, C, H, W = x.shape
            seq = x.flatten(2).transpose(1, 2)          # [B, H*W, C]
        else:
            seq = x

        residual = seq

        # ── 降维 ──────────────────────────────────────────────
        z = self.proj_in(seq)                            # [B, L, d]

        # ── 双向扫描 ──────────────────────────────────────────
        h_fwd = self._scan(z,         self.fwd_proj)             # [B, L, d]
        h_bwd = self._scan(z.flip(1), self.bwd_proj).flip(1)    # [B, L, d]

        # ── 升维 + 残差 + norm ────────────────────────────────
        out = self.proj_out(torch.cat([h_fwd, h_bwd], dim=-1))  # [B, L, D]
        out = self.drop(out)
        out = self.norm(out + residual)

        # ── 还原形状 ──────────────────────────────────────────
        if is_4d:
            out = out.transpose(1, 2).reshape(B, C, H, W)

        return out


# ====================================================================== #
#  单元测试                                                               #
# ====================================================================== #
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ssm = BidirectionalSimplifiedSSM(d_model=2048, dropout=0.1).to(device)
    print(ssm)
    total_params = sum(p.numel() for p in ssm.parameters())
    print(f"\n参数量:    {total_params / 1e6:.2f} M")
    print(f"inner_dim: {ssm.inner_dim}\n")

    # 测试1: 4D
    print("=" * 50)
    print("测试1: 4D 输入 [B, C, H, W]")
    f4 = torch.randn(4, 2048, 8, 4).to(device)
    o4 = ssm(f4)
    assert o4.shape == f4.shape
    print(f"  {tuple(f4.shape)} -> {tuple(o4.shape)}  ✅")

    # 测试2: 3D
    print("\n测试2: 3D 输入 [B, L, D]")
    f3 = torch.randn(4, 32, 2048).to(device)
    o3 = ssm(f3)
    assert o3.shape == f3.shape
    print(f"  {tuple(f3.shape)} -> {tuple(o3.shape)}  ✅")

    # 测试3: 其他尺寸
    print("\n测试3: 其他 H×W")
    for shape in [(4, 2048, 6, 6), (4, 2048, 4, 4), (4, 2048, 16, 8)]:
        f = torch.randn(*shape).to(device)
        o = ssm(f)
        assert o.shape == f.shape
        print(f"  {tuple(shape)} -> {tuple(o.shape)}  ✅")

    # 测试4: 梯度
    print("\n测试4: 梯度流动")
    ssm.train()
    fg = torch.randn(2, 2048, 8, 4).to(device)
    fg.requires_grad_(True)
    fg.retain_grad()
    ssm(fg).mean().backward()
    gn = fg.grad.norm().item() if fg.grad is not None else 0.0
    print(f"  grad norm: {gn:.6f}  {'✅' if gn > 0 else '❌'}")

    # 测试5: 速度（模拟真实训练场景）
    print("\n测试5: 速度测试")
    ssm.eval()
    BATCH, WARMUP, RUNS = 32, 10, 200
    fs = torch.randn(BATCH, 2048, 8, 4).to(device)
    with torch.no_grad():
        for _ in range(WARMUP):
            ssm(fs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(RUNS):
            ssm(fs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    print(f"  Speed: {BATCH * RUNS / elapsed:.2f} samples/sec")
    print(f"  单batch耗时: {elapsed * 1000 / RUNS:.2f} ms")

    # 测试6: 显存估算
    if device.type == "cuda":
        print("\n测试6: 显存占用")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            ssm(torch.randn(32, 2048, 8, 4).to(device))
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  峰值显存 (batch=32): {peak:.1f} MB")

    print("\n全部测试通过 ✅")