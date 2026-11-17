objects = main.o adders.o

all: $(objects)
	nvcc $(objects) -o test

%.o: %.cu
	nvcc -O3 -dc $< -o $@

clean:
	rm -f *.o test
