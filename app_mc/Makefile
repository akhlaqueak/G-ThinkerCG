NVCC := nvcc
INC := -I.. -I../common/MPMCQueue/include -I../common/bliss -I../system/ -I../common
CFLAGS := -std=c++20 -O3 -Xcompiler -fopenmp  -lgomp -lpthread -DOPENMP

BLISS_DIR = ../common
BLISS_LIB = -L${BLISS_DIR}/bliss -lbliss

all:
	nvcc -arch=sm_80 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) main.cu -o run
debug:
	nvcc -arch=sm_60 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) main.cu -g -G -Xcompiler -gdwarf-4 -o run
ap00:
	nvcc -arch=sm_60 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) main.cu -o run
