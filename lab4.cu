#include <iostream>
#include <vector>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <math.h>

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t X = call;                                           \
    if (X != cudaSuccess) {                                         \
        fprintf(stderr, "ERROR: in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(X));         \
        exit(0);                                                    \
    }                                                               \
} while(0)

struct comparator {
    __host__ __device__ bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};


__global__ void row_exchange_kernel(double* matrix, const size_t size, const int row1, const int row2) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int col = thread_id + row1; col < size + 1; col += total_threads) {
        double temp_val = matrix[col * size + row1];
        matrix[col * size + row1] = matrix[col * size + row2];
        matrix[col * size + row2] = temp_val;
    }
}

__global__ void eliminate_forward_kernel(double* matrix, const size_t size, const int current_row) {
    int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int y_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int x_stride = gridDim.x * blockDim.x;
    int y_stride = gridDim.y * blockDim.y;

    for (int row = y_idx + current_row + 1; row < size + 1; row += y_stride) {
        for (int col = x_idx + current_row + 1; col < size; col += x_stride) {
            matrix[row * size + col] -= matrix[row * size + current_row] * matrix[current_row * size + col];
        }
    }
}

__global__ void normalize_row_kernel(double* matrix, const size_t size, const int target_row) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    double pivot_element = matrix[target_row * size + target_row];

    for (int col = thread_id + target_row + 1; col < size + 1; col += total_threads) {
        matrix[col * size + target_row] /= pivot_element;
    }
}

__global__ void backward_kernel(const double* matrix, double* x, const size_t n, const int i) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for (int j = i - idx - 1; j >= 0; j -= offset) {
        x[j] -= matrix[i * n + j] * x[i];
    }
}

__host__ void solve_gaussian(double* matrix, const size_t n) {
    double* A_dev;
    CSC(cudaMalloc((void **)&A_dev, sizeof(double) * n * (n + 1)));
    CSC(cudaMemcpy(A_dev, matrix, sizeof(double) * n * (n + 1), cudaMemcpyHostToDevice));

    comparator comp;
    for (int i = 0; i < n; ++i) {
        thrust::device_ptr<double> cur_pointer = thrust::device_pointer_cast(A_dev + i * n);
        thrust::device_ptr<double> max_pointer = thrust::max_element(cur_pointer + i, cur_pointer + n, comp);
        int max_idx = max_pointer - cur_pointer;

        if (max_idx > i) {
            row_exchange_kernel<<<256, 256>>>(A_dev, n, i, max_idx);
            CSC(cudaGetLastError());
        }

        normalize_row_kernel<<<256, 256>>>(A_dev, n, i);
        CSC(cudaGetLastError());

        dim3 grid(32, 32);
        dim3 block(32, 32);
        eliminate_forward_kernel<<<grid, block>>>(A_dev, n, i);
        CSC(cudaGetLastError());    
    }

    CSC(cudaMemcpy(matrix, A_dev, sizeof(double) * n * (n + 1), cudaMemcpyDeviceToHost));

    double* x = new double[n];
    for (int i = 0; i < n; ++i){
        x[i] = matrix[n * n + i];
    }

    double* x_dev;
    CSC(cudaMalloc((void**)&x_dev, sizeof(double) * n));
    CSC(cudaMemcpy(x_dev, x, sizeof(double) * n, cudaMemcpyHostToDevice));

    for (int i = n - 1; i > 0; --i) {
        backward_kernel<<<256, 256>>>(A_dev, x_dev, n, i);
        CSC(cudaGetLastError());
    }

    CSC(cudaMemcpy(x, x_dev, sizeof(double) * n, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        std::cout << std::setprecision(10) << std::scientific << x[i] << ' ';
    }
    std::cout.put('\n');

    CSC(cudaFree(A_dev));
    CSC(cudaFree(x_dev));
    delete[] x;
}

void input_matrix_and_vector(double* A, int n) {
    // Input matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> A[j * n + i];
        }
    }
    // Input vector
    for (int i = 0; i < n; ++i) {
        std::cin >> A[n * n + i];
    }
}

int main() {
    int n;
    std::cin >> n;
    double* A = new double[n * n + n];

    input_matrix_and_vector(A, n);

    solve_gaussian(A, n);
    
    delete[] A;
}