#include <stdlib.h>
#include <cufftw.h>
#include "../convolve_fft.h"

void convolve_fft(double *signal, double *kernel, double *result, int n) {
    fftw_complex *sig_freq = (fftw_complex *)malloc(sizeof(fftw_complex) * (n/2 + 1));
    fftw_complex *ker_freq = (fftw_complex *)malloc(sizeof(fftw_complex) * (n/2 + 1));

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
    free(sig_freq);
    free(ker_freq);
}
