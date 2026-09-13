import torch

device = torch.device("cuda")
dtype = torch.bfloat16
torch.manual_seed(0)
H, eps = 896, 1e-6
w = torch.randn(H, device=device, dtype=dtype)


def reference(x, u, w):
    x = x + u
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return x, (xf * w.float()).to(x.dtype)


fused = torch.compile(reference, dynamic=False)


def bench(fn, shape, iters=200):
    x = torch.randn(*shape, device=device, dtype=dtype) * 3
    u = torch.randn(*shape, device=device, dtype=dtype)
    fn(x, u)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn(x, u)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


for shape in [(1, 1, H), (1, 512, H)]:
    x = torch.randn(*shape, device=device, dtype=dtype) * 3
    u = torch.randn(*shape, device=device, dtype=dtype)
    eager = lambda a, b: reference(a, b, w)
    comp = lambda a, b: fused(a, b, w)
    xr, nr = eager(x, u)
    xf, nf = comp(x, u)
    same = torch.equal(xr, xf) and torch.equal(nr, nf)
    print(shape, "bit-exact:", same,
          f"eager={bench(eager, shape):.4f}ms compiled={bench(comp, shape):.4f}ms")
