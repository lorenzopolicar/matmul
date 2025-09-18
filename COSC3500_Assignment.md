# COSC3500 Assignment: Parallel Programming Techniques
**Date:** August 22, 2025

> *This specification was created for the use of **46983820 Lorenzo Policar (s4698382@student.uq.edu.au)** only. Do not share this document. Sharing this document may result in a misconduct penalty.*

---

## Summary
Implement three different **matrix multiply** functions for three hardware configurations:

- **CPU (AVX/OpenMP)**
- **GPU (CUDA)**
- **MPI** on a cluster of **two nodes**

Performance is benchmarked against **Intel MKL** (CPU & MPI) or **CUBLAS** (GPU). Marks are assigned based on how fast your implementations run on matrices of size \(N \times N\). The MPI implementation may reuse your CPU-based implementation in `matrixMultiply.cpp`.

---

## Marks Table (relative runtime vs reference; lower is better)
| Marks | CPU (AVX/OpenMP) – 4 cores | GPU (CUDA) – 1 GPU | MPI – 2 nodes, 4 cores each |
|---:|:---:|:---:|:---:|
| 7 | 2.5 | 2.3 | 1.5 |
| 6 | 4.0 | 3.1 | 3.0 |
| 5 | 8.0 | 4.1 | 6.0 |
| 4 | 16.0 | 5.4 | 12.0 |
| 3 | 32.0 | 12.0 | 24.0 |
| 2 | **More than twice as slow as threshold for 3 and job does not timeout** |
| 1 | **Compiles and runs to completion and gives wrong answer** |
| 0 | **Does not compile or was not submitted or timeout** |

*Values indicate how many times longer your runtime is versus the reference. Marks are capped at 7 for each implementation.*

---

## Benchmarks
- A set of random unitary **square matrices** are successively multiplied to produce a final solution.
- The result is checked for correctness within a floating-point tolerance and measured for speed relative to **MKL/CUBLAS**.
- The benchmark repeatedly performs: \( A \leftarrow A \times B \ ).
- **CPU/GPU:** All computation is on one machine, same memory space.
- **MPI:** Each node starts with an **identical copy** of the full set of matrices. You **do not** distribute test data.  
  - Ensure all nodes **maintain a copy** of the **current product matrix** so each can compute its share of the next multiply.  
  - Each node computes its portion; all nodes must end with a **full copy** of the solution.

---

## Files You May Modify (Final Submission)
Only **three** files are permitted to change:
- `matrixMultiply.cpp`
- `matrixMultiplyGPU.cu`
- `matrixMultiplyMPI.cpp`

Constraints:
- Use only the provided headers: `matrixMultiply.h`, `matrixMultiplyGPU.cuh`, `matrixMultiplyMPI.h`.
- **Do not** write to **stdout** or to files in the final submission from your functions.

---

## Slurm Script Used for Final Grading
```
goslurm.COSC3500Assignment.RangpurJudgementDay
```
> For NVIDIA/GPU: **Do not** submit all jobs using this script (it will overwhelm the nodes). Use the provided **debug** slurm scripts for CPU, GPU, and MPI while developing and testing.

---

## Software Interface: GradeBot
`Assignment1_GradeBot` runs benchmarks and assigns marks.

**Usage:**
```bash
./Assignment1_GradeBot {matrix dimension} {threadCount} {runBenchmarkCPU} {runBenchmarkGPU} {runBenchmarkMPI} {optional integer} {optional integer} ...
```

**Examples:**
```bash
# Run CPU & MPI (no GPU), 4 threads per node, 13759x13759 matrices
./Assignment1_GradeBot 13759 4 1 0 1

# Pass extra integers to your matrixMultiply routines (for debugging/tweaking)
./Assignment1_GradeBot 13759 4 1 0 1 72 55 90 20
```

---

## Text Output
GradeBot writes to **stdout** and to per-node text files:
```
COSC3500Assignment_{benchmark type}_{node}.txt
```

Each file includes six columns:
1. **Info**: `{CPU|GPU|MPI}{mpiRank},{mpiWorldSize} | {threadCount|gpuID|mpiRank},{total physical CPU cores | number of GPUs | mpiWorldSize}({hardware run on; e.g., CPU/GPU name or node name})`
2. **N**: Matrix dimension
3. **Matrices/second (MKL)**: Reference speed (per second) using MKL (CPU/MPI) or CUBLAS (GPU)
4. **Matrices/second (You)**: Your implementation’s speed (per second)
5. **Error**: Sum of squares difference vs reference result (must be within `ERR_TOLERANCE` in `rubric.h`)
6. **Grade**: Total number of marks assigned for this result

---

## Final Submission Contents
Create a zip named **`46983820.zip`** containing:
- `matrixMultiply.cpp`
- `matrixMultiplyGPU.cu`
- `matrixMultiplyMPI.cpp`
- `slurm.zip` — an archive of all your **slurm job output files** (`slurm-xxxx.out`).

> If you did not implement a required file (e.g., only CPU implemented), include the **original blank default** `.cpp` files. The final submission **must** strictly follow this format.

---

## Getting Started
Files are available on **Rangpur** at:
```
/home/groups/cosc3500/shared/matmul/
```
Copy the `Makefile`, your three `matrixMultiply*` files, and the `/slurm/` submission scripts to your working directory (with read/write access). Then begin implementing and testing your code.

---

## Quick Checklist
- [ ] Implement CPU (`matrixMultiply.cpp`)
- [ ] Implement GPU (`matrixMultiplyGPU.cu`)
- [ ] Implement MPI (`matrixMultiplyMPI.cpp`), ensuring each node keeps a full copy of the current product matrix
- [ ] Validate numeric correctness within `ERR_TOLERANCE`
- [ ] Benchmark locally with the debug slurm scripts
- [ ] Run final benchmarks with `goslurm.COSC3500Assignment.RangpurJudgementDay`
- [ ] Package `46983820.zip` with required files + `slurm.zip`
