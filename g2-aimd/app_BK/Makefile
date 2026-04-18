NVCC := nvcc
TARGET := run
PTXAS_TARGET := run_ptxas
DEBUG_TARGET := run_debug
P100_TARGET := run_p100
PTXAS_P100_TARGET := run_ptxas_p100
DEBUG_P100_TARGET := run_debug_p100
SRC := main.cu
INC := -I.. -I../third_party/MPMCQueue/include -I../third_party/bliss-0.73 -I../system/ -I../common
CFLAGS := -std=c++20 -O3 -Xcompiler -fopenmp -lgomp -lpthread -DOPENMP
PTXASFLAGS := -Xptxas -v -lineinfo

BLISS_DIR = ../third_party/bliss-0.73
BLISS_LIB = -L${BLISS_DIR}/bliss -lbliss

.PHONY: all debug p100 debug_p100 ptxas ptxas_p100 clean

all:
	$(NVCC) -arch=sm_80 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -o $(TARGET)

ptxas:
	$(NVCC) -arch=sm_80 --maxrregcount=64 $(PTXASFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -o $(PTXAS_TARGET)

debug:
	$(NVCC) -arch=sm_80 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -g -G -Xcompiler -gdwarf-4 -o $(DEBUG_TARGET)

p100:
	$(NVCC) -arch=sm_60 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -o $(P100_TARGET)

ptxas_p100:
	$(NVCC) -arch=sm_60 --maxrregcount=64 $(PTXASFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -o $(PTXAS_P100_TARGET)

debug_p100:
	$(NVCC) -arch=sm_60 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -g -G -Xcompiler -gdwarf-4 -o $(DEBUG_P100_TARGET)

clean:
	rm -f $(TARGET) $(PTXAS_TARGET) $(DEBUG_TARGET) $(P100_TARGET) $(PTXAS_P100_TARGET) $(DEBUG_P100_TARGET)
