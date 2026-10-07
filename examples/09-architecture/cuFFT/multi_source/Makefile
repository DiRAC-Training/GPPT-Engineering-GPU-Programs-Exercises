CC ?= gcc
CFLAGS = -O2 -Wall
LIBS = -lm

ifdef USE_GPU
  CFLAGS += -I$(NVHPC_ROOT)/cuda/include -I$(NVHPC_ROOT)/math_libs/include
  SOURCES = src/main.c src/gpu/convolve_fft.c
  LIBS += -L$(NVHPC_ROOT)/cuda/lib64 -L$(NVHPC_ROOT)/math_libs/lib64 -lcufftw -lcufft
else
  SOURCES = src/main.c src/cpu/convolve_fft.c
  LIBS += -lfftw3
endif

signal_smooth: $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $(LIBS)

clean:
	rm -f signal_smooth signal_output.dat

.PHONY: clean
