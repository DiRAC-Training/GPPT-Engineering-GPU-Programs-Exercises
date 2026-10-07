all: run

%: %.cpp
	nvc++ -stdpar=gpu $< -o $@

run: main
	./main
