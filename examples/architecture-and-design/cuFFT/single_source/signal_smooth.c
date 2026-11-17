#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef USE_GPU
#include <cufftw.h>
#else
#include <fftw3.h>
#endif

#define N (1 << 22)  /* 4,194,304 samples */
#define KERNEL_WIDTH 127 /*Gaussian kernel width*/

// Forward declarations (see end of file)
double get_time_sec(void);
void generate_noisy_signal(double *signal, int n);
void create_gaussian_kernel(double *kernel, int n, int width, double sigma);
void write_output(const char *filename, double *signal, double *result, int n);

void convolve_fft(double *signal, double *kernel, double *result, int n) {
#ifdef USE_GPU
    fftw_complex *sig_freq = (fftw_complex *)malloc(sizeof(fftw_complex) * (n/2 + 1));
    fftw_complex *ker_freq = (fftw_complex *)malloc(sizeof(fftw_complex) * (n/2 + 1));
#else
    fftw_complex *sig_freq = fftw_alloc_complex(n/2 + 1);
    fftw_complex *ker_freq = fftw_alloc_complex(n/2 + 1);
#endif

    fftw_plan fwd_signal = fftw_plan_dft_r2c_1d(n, signal, sig_freq, FFTW_ESTIMATE);
    fftw_plan fwd_kernel = fftw_plan_dft_r2c_1d(n, kernel, ker_freq, FFTW_ESTIMATE);
    fftw_plan inv_result = fftw_plan_dft_c2r_1d(n, sig_freq, result, FFTW_ESTIMATE);

    fftw_execute(fwd_signal);
    fftw_execute(fwd_kernel);

    for (int i = 0; i < n/2 + 1; i++) {
        double re = sig_freq[i][0] * ker_freq[i][0] - sig_freq[i][1] * ker_freq[i][1];
        double im = sig_freq[i][0] * ker_freq[i][1] + sig_freq[i][1] * ker_freq[i][0];
        sig_freq[i][0] = re;
        sig_freq[i][1] = im;
    }

    fftw_execute(inv_result);

    for (int i = 0; i < n; i++) result[i] /= n;

    fftw_destroy_plan(fwd_signal);
    fftw_destroy_plan(fwd_kernel);
    fftw_destroy_plan(inv_result);
#ifdef USE_GPU
    free(sig_freq);
    free(ker_freq);
#else
    fftw_free(sig_freq);
    fftw_free(ker_freq);
#endif
}

int main(void) {
    srand(42);  /* Fixed seed for reproducibility */

#ifdef USE_GPU
    double *signal = (double *)malloc(sizeof(double) * N);
    double *kernel = (double *)malloc(sizeof(double) * N);
    double *result = (double *)malloc(sizeof(double) * N);
#else
    double *signal = fftw_alloc_real(N);
    double *kernel = fftw_alloc_real(N);
    double *result = fftw_alloc_real(N);
#endif

    generate_noisy_signal(signal, N);
    create_gaussian_kernel(kernel, N, KERNEL_WIDTH, 2.0);

    double t_start = get_time_sec();
    convolve_fft(signal, kernel, result, N);
    double t_end = get_time_sec();

    printf("FFT convolution took %.4f seconds (N = %d)\n", t_end - t_start, N);

    write_output("signal_output.dat", signal, result, N);

#ifdef USE_GPU
    free(signal);
    free(kernel);
    free(result);
#else
    fftw_free(signal);
    fftw_free(kernel);
    fftw_free(result);
#endif

    return 0;
}

// YOU CAN IGNORE BELOW THIS

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void generate_noisy_signal(double *signal, int n) {
    for (int i = 0; i < n; i++) {
        double x = (double)i / n * 4.0 * M_PI;
        signal[i] = sin(x) + 0.3 * ((double)rand() / RAND_MAX - 0.5);
    }
}

void create_gaussian_kernel(double *kernel, int n, int width, double sigma) {
    for (int i = 0; i < n; i++) kernel[i] = 0.0;

    double sum = 0.0;
    for (int i = 0; i < width; i++) {
        int idx = (i - width/2 + n) % n;
        double x = i - width/2;
        kernel[idx] = exp(-x*x / (2*sigma*sigma));
        sum += kernel[idx];
    }

    for (int i = 0; i < n; i++) kernel[i] /= sum;
}

void write_output(const char *filename, double *signal, double *result, int n) {
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%d %f %f\n", i, signal[i], result[i]);
    }
    fclose(f);
}
