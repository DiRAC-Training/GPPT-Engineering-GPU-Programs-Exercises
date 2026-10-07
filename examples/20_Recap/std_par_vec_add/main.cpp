#include <random>
#include <ctime>
#include <algorithm>
#include <execution>

const int N = 32 << 20;
typedef double real;
const real epsilon = 0.00001;

void initArray(real* A, int length) {
     std::srand(std::time({}));
    for(int i=0; i<length; i++) {
        A[i] = rand() / (real)RAND_MAX;
    }
}

int main() {
    real* A = new real[N];
    real* B = new real[N];
    real* C = new real[N];

    initArray(A, N);
    initArray(B, N);

    std::transform(std::execution::par_unseq, A, A + N, B, C,
        [](real a, real b){ return a + b; });

    bool results_match = true;
    for(int i=0; i < N; ++i) {
        const real cpu_result = A[i] + B[i];
        if(fabs(C[i] - cpu_result) > epsilon) {
            printf("Index %d mismatch: %f != %f", i, C[i], cpu_result);
            results_match = false;
            break;
        }
    }
    if(results_match) printf("Results match!");
}
