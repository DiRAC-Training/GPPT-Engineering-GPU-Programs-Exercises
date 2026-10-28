CC ?= gcc
CFLAGS = -O2 -Wall
LIBS = -lm

CUDA_ROOT=$(CUDA_PATH)
# CUDA_ROOT=$(NVHPC_ROOT)/cuda

ifdef USE_GPU
  CFLAGS += -DUSE_GPU -I$(CUDA_ROOT)/include -I$(CUDA_ROOT)/math_libs/include
  LIBS += -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/math_libs/lib64 -lcufftw -lcufft
else
  LIBS += -lfftw3
endif

EXES=signal_smooth signal_smooth_refactored_malloc

all: $(EXES)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(EXES) signal_output.dat

.PHONY: clean
