#include <matrixMultiply.h>
#include <omp.h>
#include <immintrin.h>
#include <cstring>
#include <algorithm>
#define STUDENTID 46983820 //DO NOT REMOVE

// AVX vectorization helper functions
inline void avx_multiply_add(floatType* C, const floatType* B, floatType a_val, int N, int j_start, int j_end) {
    // For floatType (assuming float), use AVX
    if constexpr (sizeof(floatType) == 4) { // float
        __m256 a_vec = _mm256_set1_ps(a_val);
        int j = j_start;
        
        // Handle unaligned elements at the beginning
        while (j < j_end && (reinterpret_cast<uintptr_t>(&C[j]) % 32 != 0)) {
            C[j] += a_val * B[j];
            j++;
        }
        
        // Vectorized loop for aligned elements
        for (; j < j_end - 7; j += 8) {
            __m256 b_vec = _mm256_loadu_ps(&B[j]);
            __m256 c_vec = _mm256_loadu_ps(&C[j]);
            __m256 result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            _mm256_storeu_ps(&C[j], result);
        }
        
        // Handle remaining elements
        for (; j < j_end; j++) {
            C[j] += a_val * B[j];
        }
    } else { // double
        __m256d a_vec = _mm256_set1_pd(a_val);
        int j = j_start;
        
        // Handle unaligned elements at the beginning
        while (j < j_end && (reinterpret_cast<uintptr_t>(&C[j]) % 32 != 0)) {
            C[j] += a_val * B[j];
            j++;
        }
        
        // Vectorized loop for aligned elements
        for (; j < j_end - 3; j += 4) {
            __m256d b_vec = _mm256_loadu_pd(&B[j]);
            __m256d c_vec = _mm256_loadu_pd(&C[j]);
            __m256d result = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
            _mm256_storeu_pd(&C[j], result);
        }
        
        // Handle remaining elements
        for (; j < j_end; j++) {
            C[j] += a_val * B[j];
        }
    }
}

/**
* @brief Implements an NxN matrix multiply C=A*B
*  		   	 	 	 			  		  		 	      
* @param[in] N : dimension of square matrix (NxN)
* @param[in] A : pointer to input NxN matrix
* @param[in] B : pointer to input NxN matrix
* @param[out] C : pointer to output NxN matrix
* @param[in] args : pointer to array of integers which can be used for debugging and performance tweaks. Optional. If unused, set to zero
* @param[in] argCount : the length of the flags array
* @return : your student ID
*  		   	 	 	 			  		  		 	      
* */
int matrixMultiply(int N, const floatType* A, const floatType* B, floatType* C, int* args, int argCount){  		   	 	 	 			  		  		 	      
    if (N<=0) { return STUDENTID;}//Your code must be able to deal with N=0 scenario without crashing.  		   	 	 	 			  		  		 	      

    // Initialize output matrix to zero
    std::memset(C, 0, N * N * sizeof(floatType));
    
    // Get block size from args if provided, otherwise use default
    int blockSize = (args && argCount > 0) ? args[0] : 64;
    if (blockSize <= 0) blockSize = 64;
    
    // Ensure block size doesn't exceed matrix size
    if (blockSize > N) blockSize = N;
    
    // For very small matrices, use simple implementation
    if (N < 32) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                floatType a_ik = A[i * N + k];
                for (int j = 0; j < N; j++) {
                    C[i * N + j] += a_ik * B[k * N + j];
                }
            }
        }
        return STUDENTID;
    }
    
    // Blocked matrix multiplication with OpenMP parallelization
    // Use dynamic scheduling for better load balancing
    #pragma omp parallel for schedule(dynamic, 1)
    for (int ii = 0; ii < N; ii += blockSize) {
        for (int jj = 0; jj < N; jj += blockSize) {
            for (int kk = 0; kk < N; kk += blockSize) {
                // Process block
                int i_end = std::min(ii + blockSize, N);
                int j_end = std::min(jj + blockSize, N);
                int k_end = std::min(kk + blockSize, N);
                
                // Optimize loop order for better cache locality
                for (int k = kk; k < k_end; k++) {
                    for (int i = ii; i < i_end; i++) {
                        floatType a_ik = A[i * N + k];
                        // Use AVX vectorization for the inner loop
                        avx_multiply_add(&C[i * N + jj], &B[k * N + jj], a_ik, N, jj, j_end);
                    }
                }
            }
        }
    }

    return STUDENTID;  		   	 	 	 			  		  		 	      
}
  		   	 	 	 			  		  		 	      
