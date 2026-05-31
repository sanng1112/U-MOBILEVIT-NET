"""
Thực nghiệm cơ chế: đo tốc độ suy giảm hạng hiệu dụng (rank collapse) của
linear attention baseline so với INLA khi xếp chồng nhiều lớp.

Thiết kế (theo tinh thần Dong et al., 2021 — pure attention mất hạng theo độ sâu):
áp dụng liên tiếp L khối attention (không residual, không MLP — để cô lập tác động
của riêng phép aggregation), đo effective rank của ma trận biểu diễn token (N×d)
ở mỗi độ sâu. Lặp qua nhiều seed và lấy trung bình.

Chạy: PYTHONPATH=. python tools/exp_rank_collapse.py
"""
import argparse
import torch

from cv_nets.blocks.inla import INLAAttention, INLATransformerBlock
from cv_nets.utils.spectral import effective_rank, spectral_entropy, singular_value_decay


def _correlated_input(n_tokens, dim, rank, seed):
    """Sinh token có hạng nội tại thấp (mô phỏng đặc trưng đã qua vài lớp conv)."""
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(rank, dim, generator=g)
    coeff = torch.randn(n_tokens, rank, generator=g)
    x = coeff @ basis
    x = x + 0.05 * torch.randn(n_tokens, dim, generator=g)   # nhiễu nhẹ
    return x.unsqueeze(0)


@torch.no_grad()
def run_stack(dim, n_tokens, depth, use_lifting, dim_expand, seed, mode="block"):
    torch.manual_seed(seed)
    x = _correlated_input(n_tokens, dim, rank=min(dim, 32), seed=seed)
    if mode == "block":   # kiến trúc thực tế: residual + MLP
        layers = [INLATransformerBlock(dim, dim_expand=dim_expand, use_lifting=use_lifting).eval()
                  for _ in range(depth)]
    else:                 # attention thuần (cô lập aggregation)
        layers = [INLAAttention(dim, dim_expand=dim_expand, use_lifting=use_lifting).eval()
                  for _ in range(depth)]
    eranks, sents, top1s = [], [], []
    z = x
    for layer in layers:
        z = layer(z)
        eranks.append(effective_rank(z[0]))
        sents.append(spectral_entropy(z[0]))
        top1s.append(singular_value_decay(z[0])[1])
    return eranks, sents, top1s


def average(dim, n_tokens, depth, use_lifting, dim_expand, seeds, mode="block"):
    accE = [0.0] * depth; accS = [0.0] * depth; accT = [0.0] * depth
    for s in seeds:
        e, se, t = run_stack(dim, n_tokens, depth, use_lifting, dim_expand, s, mode)
        for i in range(depth):
            accE[i] += e[i]; accS[i] += se[i]; accT[i] += t[i]
    n = len(seeds)
    return ([v / n for v in accE], [v / n for v in accS], [v / n for v in accT])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--tokens", type=int, default=196)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--dim-expand", type=int, default=128)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--mode", type=str, default="block", choices=["block", "attn"],
                   help="block = transformer (residual+MLP, thực tế); attn = attention thuần")
    args = p.parse_args()
    seeds = list(range(args.seeds))

    print(f"# Rank-collapse [{args.mode}]: dim={args.dim}, tokens={args.tokens}, depth={args.depth}, "
          f"dim_expand(r)={args.dim_expand}, seeds={args.seeds}")
    print(f"# effective_rank tối đa lý thuyết = min(N,d) = {min(args.tokens, args.dim)}\n")

    baseE, baseS, baseT = average(args.dim, args.tokens, args.depth, False, args.dim_expand, seeds, args.mode)
    inlaE, inlaS, inlaT = average(args.dim, args.tokens, args.depth, True, args.dim_expand, seeds, args.mode)

    print(f"{'depth':>5} | {'erank_base':>10} {'erank_INLA':>10} {'Δ%':>7} | "
          f"{'sEnt_base':>9} {'sEnt_INLA':>9} | {'top1_base':>9} {'top1_INLA':>9}")
    print("-" * 88)
    for i in range(args.depth):
        d = i + 1
        dpct = 100.0 * (inlaE[i] - baseE[i]) / max(baseE[i], 1e-9)
        print(f"{d:>5} | {baseE[i]:>10.2f} {inlaE[i]:>10.2f} {dpct:>6.1f}% | "
              f"{baseS[i]:>9.3f} {inlaS[i]:>9.3f} | {baseT[i]:>9.3f} {inlaT[i]:>9.3f}")

    print("\n# Tóm tắt tại lớp cuối (depth={}):".format(args.depth))
    print(f"#   effective rank:  baseline={baseE[-1]:.2f}  INLA={inlaE[-1]:.2f}  "
          f"(+{100*(inlaE[-1]-baseE[-1])/max(baseE[-1],1e-9):.1f}%)")
    print(f"#   spectral entropy: baseline={baseS[-1]:.3f}  INLA={inlaS[-1]:.3f}")
    print(f"#   top-1 energy:     baseline={baseT[-1]:.3f}  INLA={inlaT[-1]:.3f} (thấp hơn = phổ ít dốc hơn)")


if __name__ == "__main__":
    main()
