CC ?= gcc
CFLAGS = -O2 -Wall
LIBS = -lm

ifdef USE_GPU
  CFLAGS += -DUSE_GPU -I$(NVHPC_ROOT)/cuda/include -I$(NVHPC_ROOT)/math_libs/include
  LIBS += -L$(NVHPC_ROOT)/cuda/lib64 -L$(NVHPC_ROOT)/math_libs/lib64 -lcufftw -lcufft
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
