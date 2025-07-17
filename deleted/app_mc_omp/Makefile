CCOMPILE=g++
CPPFLAGS= -O3

all: 
	$(CCOMPILE) -std=c++20 -fopenmp -lpthread run.cpp $(CPPFLAGS)  -o run
debug: 
	$(CCOMPILE) -std=c++20 -fopenmp -lpthread run.cpp -O0 -o run -g -gdwarf-4

clean:
	-rm run
