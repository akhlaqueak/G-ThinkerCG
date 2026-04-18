NVCC := nvcc
INC := -I.. -I../third_party/MPMCQueue/include -I../third_party/bliss-0.73 -I../system/ -I../common
CFLAGS := -std=c++20 -O3 -Xcompiler -fopenmp -lgomp -lpthread -DOPENMP
PTXASFLAGS := -Xptxas -v -lineinfo

BLISS_DIR = ../third_party/bliss-0.73
BLISS_LIB = -L${BLISS_DIR}/bliss -lbliss

all:
	nvcc -arch=sm_80 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) main.cu -o run
ptxas:
	nvcc -arch=sm_80 --maxrregcount=64 $(PTXASFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) main.cu -o run
debug:
	nvcc -arch=sm_80 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) main.cu -g -G -Xcompiler -gdwarf-4 -o run

p100:
	nvcc -arch=sm_60 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) main.cu -o run
ptxas_p100:
	nvcc -arch=sm_60 --maxrregcount=64 $(PTXASFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) main.cu -o run
debug_p100:
	nvcc -arch=sm_60 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) main.cu -g -G -Xcompiler -gdwarf-4 -o run
