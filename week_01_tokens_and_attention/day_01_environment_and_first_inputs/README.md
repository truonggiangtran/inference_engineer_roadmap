# Day 1: Environment and Your First Model Inputs

**Time:** 60 minutes

## Goal

Establish a reproducible environment and understand how Python values become tensors.

## Knowledge

**Why record the environment?** Results from inference experiments depend on GPU, driver, CUDA runtime, PyTorch, and Transformers versions. A benchmark result without them can't be reproduced or compared.

**Two different CUDA versions.** PyTorch wheels bundle their own CUDA runtime (`torch.version.cuda`). The CUDA toolkit you install locally (`nvcc --version`) is only used to compile your own kernels (Day 20). They can differ. The NVIDIA driver must be new enough for both.

**`sm_120`.** The RTX 5070 is a Blackwell consumer GPU with compute capability 12.0. A PyTorch build only runs native kernels on your GPU if its compiled architecture list (`torch.cuda.get_arch_list()`) includes `sm_120`. Older wheels may install fine but then fail with "no kernel image is available".

**Tensors.** A tensor is an n-dimensional array with three properties you will check constantly:

| Attribute | Meaning | Example |
| --- | --- | --- |
| `shape` | Size of each dimension | `torch.Size([1, 5])` = 1 sequence of 5 tokens |
| `dtype` | Element type | `torch.int64` for token IDs, `torch.bfloat16` for weights |
| `device` | Where the data lives | `cpu` or `cuda:0` |

Token IDs are integers that index the vocabulary, so they use an integer dtype, normally `int64` (`torch.long`). Most operations require all inputs to be on the same device. Moving data with `.to("cuda")` copies it across PCIe.

**Asynchronous GPU work.** CUDA operations are queued and return immediately. `torch.cuda.synchronize()` waits until queued work has finished. Reading a value back to Python with `.item()` or `.cpu()` also synchronizes. Day 16 builds on this.

## References

- [PyTorch installation selector](https://pytorch.org/get-started/locally/): choose Linux, pip, and a CUDA version whose wheels support `sm_120`.
- [Python introduction](https://docs.python.org/3/tutorial/introduction.html): numbers, strings, and lists.
- [PyTorch tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html): creating tensors, attributes, and moving data to a GPU.
- [NVIDIA compute capabilities](https://developer.nvidia.com/cuda-gpus): confirm the RTX 5070's compute capability.

## Practice

Complete [exercise.py](exercise.py):

1. `environment_report()`: collect the Python, PyTorch, bundled CUDA runtime, Transformers, GPU name, compute capability, PyTorch architecture list, driver, and local `nvcc` versions.
2. `token_ids_to_tensor()`: convert a Python list of token IDs into an `int64` tensor shaped `[1, seq_len]`.
3. `gpu_smoke_test()`: move a small tensor to the GPU, run an operation, synchronize, and return the result. If CUDA is unavailable, return a clear blocker message instead.

```bash
python week_01_tokens_and_attention/day_01_environment_and_first_inputs/exercise.py
```

The script writes `results/environment.json`. Keep this file because Day 27 uses it.

## Done when

You can explain a tensor's shape, dtype, and device, and you have either a successful GPU check or a clearly recorded setup blocker.
