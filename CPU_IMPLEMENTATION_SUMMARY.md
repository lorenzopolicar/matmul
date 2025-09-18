# CPU Matrix Multiplication Implementation Summary

## Overview
This document summarizes the optimized CPU matrix multiplication implementation for the COSC3500 assignment.

## Key Optimizations Implemented

### 1. **Blocking/Tiling Strategy**
- **Block Size**: Configurable via `args[0]` parameter (default: 64x64)
- **Cache Optimization**: Blocks fit in L1/L2 cache for better memory access patterns
- **Three-level blocking**: Outer loops iterate over blocks, inner loops process elements

### 2. **OpenMP Parallelization**
- **Parallel Region**: `#pragma omp parallel for schedule(dynamic, 1)`
- **Dynamic Scheduling**: Better load balancing across 4 cores
- **Thread Safety**: Each thread processes different blocks independently

### 3. **AVX Vectorization**
- **SIMD Operations**: Uses AVX2 intrinsics for 8x float or 4x double operations
- **FMA Instructions**: `_mm256_fmadd_ps/pd` for fused multiply-add operations
- **Memory Alignment**: Handles both aligned and unaligned memory access
- **Fallback**: Scalar operations for remaining elements

### 4. **Memory Access Optimization**
- **Loop Reordering**: k-i-j order for better cache locality
- **Data Reuse**: Maximizes reuse of A[i,k] across j iterations
- **Cache-Friendly**: Blocking ensures data stays in cache

### 5. **Special Cases**
- **Small Matrices**: Simple implementation for N < 32
- **Edge Cases**: Proper handling of N=0 and non-divisible block sizes
- **Boundary Handling**: `std::min()` for block boundaries

## Implementation Details

### Function Signature
```cpp
int matrixMultiply(int N, const floatType* A, const floatType* B, floatType* C, int* args, int argCount)
```

### Parameters
- `N`: Matrix dimension (N×N)
- `A, B`: Input matrices (row-major order)
- `C`: Output matrix (initialized to zero)
- `args`: Optional parameters (args[0] = block size)
- `argCount`: Number of arguments provided

### Algorithm Structure
1. **Input Validation**: Handle N ≤ 0 case
2. **Matrix Initialization**: Zero out output matrix C
3. **Block Size Configuration**: Use args[0] or default (64)
4. **Small Matrix Optimization**: Simple loop for N < 32
5. **Blocked Computation**: Three-level nested loops with AVX

### AVX Vectorization Details
```cpp
// For float (4 bytes)
__m256 a_vec = _mm256_set1_ps(a_val);  // Broadcast scalar
__m256 b_vec = _mm256_loadu_ps(&B[j]); // Load 8 floats
__m256 c_vec = _mm256_loadu_ps(&C[j]); // Load 8 floats
__m256 result = _mm256_fmadd_ps(a_vec, b_vec, c_vec); // FMA operation
_mm256_storeu_ps(&C[j], result);       // Store 8 floats

// For double (8 bytes)
__m256d a_vec = _mm256_set1_pd(a_val);  // Broadcast scalar
__m256d b_vec = _mm256_loadu_pd(&B[j]); // Load 4 doubles
__m256d c_vec = _mm256_loadu_pd(&C[j]); // Load 4 doubles
__m256d result = _mm256_fmadd_pd(a_vec, b_vec, c_vec); // FMA operation
_mm256_storeu_pd(&C[j], result);        // Store 4 doubles
```

## Performance Characteristics

### Expected Performance
- **Target**: <2.5x slower than Intel MKL
- **Optimization Level**: High (blocking + AVX + OpenMP)
- **Memory Bandwidth**: Optimized for cache hierarchy
- **Parallel Efficiency**: Good load balancing with dynamic scheduling

### Scalability
- **Thread Scaling**: Up to 4 cores (assignment requirement)
- **Matrix Size**: Efficient for large matrices (N > 64)
- **Block Size**: Tunable via args parameter

## Testing Strategy

### Debug Testing
```bash
# Use debug slurm script
sbatch slurm/goslurm_COSC3500Assignment_RangpurDebugCPU
```

### Performance Tuning
- **Block Size**: Test with 32, 64, 128, 256
- **Thread Count**: Verify 4-core utilization
- **Matrix Sizes**: Test with 128, 512, 1024, 2048, 4096

### Correctness Validation
- **Numerical Accuracy**: Compare with reference implementation
- **Edge Cases**: N=0, N=1, small matrices
- **Memory Safety**: No buffer overflows or memory leaks

## Compilation Requirements

### Compiler Flags
```bash
g++ -std=c++11 -O2 -mavx -fopenmp -c matrixMultiply.cpp
```

### Dependencies
- **OpenMP**: For parallelization
- **AVX2**: For vectorization (Intel/AMD x86_64)
- **Standard Library**: `<immintrin.h>`, `<omp.h>`, `<cstring>`, `<algorithm>`

## Future Optimizations

### Potential Improvements
1. **AVX-512**: If available on target hardware
2. **Prefetching**: Manual prefetch instructions
3. **Loop Unrolling**: More aggressive unrolling
4. **Memory Alignment**: Aligned memory allocation
5. **NUMA Awareness**: For multi-socket systems

### Tuning Parameters
- **Block Size**: Optimize for specific cache sizes
- **Thread Affinity**: Bind threads to specific cores
- **Memory Layout**: Consider column-major for B matrix

## Conclusion

This implementation combines multiple optimization techniques to achieve high performance:
- **Blocking** for cache efficiency
- **AVX vectorization** for SIMD operations
- **OpenMP parallelization** for multi-core utilization
- **Memory access optimization** for better bandwidth utilization

The implementation is designed to be competitive with Intel MKL while maintaining correctness and robustness across different matrix sizes and edge cases.
