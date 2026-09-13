import torch
import torch.nn.functional as F

device = torch.device("cuda")
dtype = torch.bfloat16
torch.manual_seed(0)


def has_enable_gqa() -> bool:
    try:
        F.scaled_dot_product_attention(
            torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4),
            enable_gqa=True,
        )
        return True
    except (TypeError, RuntimeError):
        return False


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


print("enable_gqa available:", has_enable_gqa())
rows = []
for label, qh, kvh in [("TP=1 (14Q/2KV)", 14, 2), ("TP=2 (7Q/1KV)", 7, 1)]:
    for T in (1, 128, 512, 2048):
        q = torch.randn(1, qh, T, 64, device=device, dtype=dtype)
        k = torch.randn(1, kvh, T, 64, device=device, dtype=dtype)
        v = torch.randn(1, kvh, T, 64, device=device, dtype=dtype)

        g = qh // kvh
        if kvh == 1:
            k_m = k.expand(1, qh, T, 64)
            v_m = v.expand(1, qh, T, 64)
        else:
            k_m = k.repeat_interleave(g, dim=1)
            v_m = v.repeat_interleave(g, dim=1)

        def manual():
            return F.scaled_dot_product_attention(q, k_m, v_m, is_causal=(T > 1))

        def native():
            return F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1), enable_gqa=True)

        d = (manual().float() - native().float()).abs().max().item()
        rows.append((label, T, bench(manual), bench(native), d))
for r in rows:
    print(f"{r[0]:<16} T={r[1]:<5} manual={r[2]:.4f}ms native_gqa={r[3]:.4f}ms maxdiff={r[4]:.2e}")
