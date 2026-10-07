#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <thrust/random.h>

const int N = 32 << 20;
const float epsilon = 0.00001;

typedef double real;

int main() {
    // Generate random data serially.
    thrust::default_random_engine rng(42);
    thrust::uniform_real_distribution<real> dist(-50.0, 50.0);

    thrust::host_vector<real> h_A(N);
    thrust::host_vector<real> h_B(h_A.size());
    thrust::generate(h_A.begin(), h_A.end(), [&] { return dist(rng); });
    thrust::generate(h_B.begin(), h_B.end(), [&] { return dist(rng); });

    // Transfer to device and compute the sum.
    thrust::device_vector<real> d_A = h_A;
    thrust::device_vector<real> d_B = h_B;
    thrust::device_vector<real> d_C(d_A.size());
    thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_C.begin(), cuda::std::plus<real>());

    thrust::host_vector<real> h_C = d_C;

    bool results_match = true;
    for(int i=0; i < h_C.size(); ++i) {
        const float cpu_result = h_A[i] + h_B[i];
        if(fabs(h_C[i] - cpu_result) > epsilon) {
            printf("Index %d mismatch: %f != %f", i, h_C[i], cpu_result);
            results_match = false;
            break;
        }
    }
    if(results_match) printf("Results match!");
}
