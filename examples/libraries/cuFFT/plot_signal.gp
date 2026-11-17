set terminal pngcairo size 800,400
set output 'signal_smooth.png'

set title 'FFT-based Signal Smoothing'
set xlabel 'Sample'
set ylabel 'Amplitude'
set grid

plot 'signal_output.dat' using 1:2 with lines title 'Noisy signal' lc rgb '#cc0000', \
     'signal_output.dat' using 1:3 with lines title 'Smoothed' lc rgb '#0000cc' lw 2
