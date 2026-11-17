#ifndef CONVOLVE_FFT_H
#define CONVOLVE_FFT_H

/**
 * @brief Performs convolution of two signals using FFT.
 *
 * @param[in]  signal  Input signal array of size n.
 * @param[in]  kernel  Convolution kernel array of size n.
 * @param[out] result  Output array of size n for the convolved signal.
 * @param[in]  n       Size of all arrays (must be the same).
 */
void convolve_fft(double *signal, double *kernel, double *result, int n);

#endif /* CONVOLVE_FFT_H */
