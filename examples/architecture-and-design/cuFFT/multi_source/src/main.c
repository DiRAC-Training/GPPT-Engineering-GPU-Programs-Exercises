#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "convolve_fft.h"

#define N (1 << 22)  /* 4,194,304 samples */
#define KERNEL_WIDTH 127 /*Gaussian kernel width*/

/**
 * @brief Returns the current wall-clock time in seconds.
 *
 * Uses CLOCK_MONOTONIC for consistent timing measurements.
 *
 * @return Current time in seconds (with nanosecond precision).
 */
double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/**
 * @brief Generates a test signal: sine wave with additive noise.
 *
 * Creates a signal containing two complete sine wave periods with
 * uniformly distributed random noise added.
 *
 * @param[out] signal  Pre-allocated array to store the generated signal.
 * @param[in]  n       Number of samples to generate.
 */
void generate_noisy_signal(double *signal, int n) {
    for (int i = 0; i < n; i++) {
        double x = (double)i / n * 4.0 * M_PI;
        signal[i] = sin(x) + 0.3 * ((double)rand() / RAND_MAX - 0.5);
    }
}

/**
 * @brief Creates a normalized Gaussian smoothing kernel.
 *
 * The kernel is centered and wrapped around for use with FFT-based
 * circular convolution. Zero-padding fills the remainder of the array.
 * The kernel is normalized so that its values sum to 1.
 *
 * @param[out] kernel  Pre-allocated array of size n (will be zero-padded).
 * @param[in]  n       Total size of the kernel array (must match signal size).
 * @param[in]  width   Number of non-zero samples in the Gaussian.
 * @param[in]  sigma   Standard deviation of the Gaussian.
 */
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

/**
 * @brief Writes signal data to a file for plotting.
 *
 * Output format: "index original_value smoothed_value" per line.
 * Suitable for plotting with gnuplot or similar tools.
 *
 * @param[in] filename  Path to the output file.
 * @param[in] signal    Original input signal.
 * @param[in] result    Smoothed output signal.
 * @param[in] n         Number of samples.
 */
void write_output(const char *filename, double *signal, double *result, int n) {
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%d %f %f\n", i, signal[i], result[i]);
    }
    fclose(f);
}

int main(void) {
    srand(42);  /* Fixed seed for reproducibility */

    double *signal = (double *)malloc(sizeof(double) * N);
    double *kernel = (double *)malloc(sizeof(double) * N);
    double *result = (double *)malloc(sizeof(double) * N);

    generate_noisy_signal(signal, N);
    create_gaussian_kernel(kernel, N, KERNEL_WIDTH, 2.0);

    double t_start = get_time_sec();
    convolve_fft(signal, kernel, result, N);
    double t_end = get_time_sec();

    printf("FFT convolution took %.4f seconds (N = %d)\n", t_end - t_start, N);

    write_output("signal_output.dat", signal, result, N);

    free(signal);
    free(kernel);
    free(result);

    return 0;
}
