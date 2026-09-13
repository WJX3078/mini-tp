# TP Topology (measured on the development host)

| item | value |
|---|---|
| GPU | `NVIDIA GeForce RTX 4060 Laptop GPU` |
| GPU count | 1 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| NCCL compiled | False |

```
ERROR: Option -m is missing its value. Please run 'nvidia-smi -h' for help.
```

## Interpretation

- This is a **single-GPU, PCIe-attached laptop** host. There is no NVLink and
  no second GPU, so every TP>1 GPU number in the README is **UNVERIFIED**;
  the CPU/Gloo collective floor (docs/TP_SCALING.md) is the only measured
  communication data.
- NCCL small-message latency (the 1792 B decode AllReduce) is dominated by
  launch/synchronization, not bandwidth: expect NVLink domains to hide
  multi-µs AllReduces while PCIe halves effective bandwidth
  (~16-32 GB/direction per docs/COMMUNICATION_PROFILING.md). Real numbers
  require the multi-GPU host — `tests/test_multi_gpu.py` runs automatically
  there.
- The torch build on this host has **no NCCL compiled**
  (`dist.is_nccl_available() == False`) and the torchrun agent is broken on
  this Windows setup; both facts are recorded by benchmark metadata
  (`nccl_available`, `comm_backend`) so NCCL-absent hosts are
  self-describing.
