all: run

%: %.cpp
	nvc++ -stdpar=gpu thrust_vec_add.cpp -o thrust_vec_add

run: thrust_vec_add
	./thrust_vec_add
