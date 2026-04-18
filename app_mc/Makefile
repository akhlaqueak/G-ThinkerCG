NVCC := nvcc
TARGET := run
PTXAS_TARGET := run_ptxas
DEBUG_TARGET := run_debug
P100_TARGET := run_p100
PTXAS_P100_TARGET := run_ptxas_p100
DEBUG_P100_TARGET := run_debug_p100
SRC := main.cu
INC := -I.. -I../common/MPMCQueue/include -I../common/bliss -I../system/ -I../common
CFLAGS := -std=c++20 -O3 -Xcompiler -fopenmp  -lgomp -lpthread -DOPENMP
PTXASFLAGS := -Xptxas -v -lineinfo

BLISS_DIR = ../common
BLISS_LIB = -L${BLISS_DIR}/bliss -lbliss

.PHONY: all debug p100 debug_p100 ptxas ptxas_p100 clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -arch=sm_80 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) $< -o $@

ptxas: $(PTXAS_TARGET)

$(PTXAS_TARGET): $(SRC)
	$(NVCC) -arch=sm_80 --maxrregcount=64 $(PTXASFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $< -o $@

debug: $(DEBUG_TARGET)

$(DEBUG_TARGET): $(SRC)
	$(NVCC) -arch=sm_80 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) $< -g -G -Xcompiler -gdwarf-4 -o $@

p100: $(P100_TARGET)

$(P100_TARGET): $(SRC)
	$(NVCC) -arch=sm_60 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) $< -o $@

ptxas_p100: $(PTXAS_P100_TARGET)

$(PTXAS_P100_TARGET): $(SRC)
	$(NVCC) -arch=sm_60 --maxrregcount=64 $(PTXASFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $< -o $@

debug_p100: $(DEBUG_P100_TARGET)

$(DEBUG_P100_TARGET): $(SRC)
	$(NVCC) -arch=sm_60 --maxrregcount=64 $(CFLAGS) $(INC) $(BLISS_LIB) $< -g -G -Xcompiler -gdwarf-4 -o $@

clean:
	rm -f $(TARGET) $(PTXAS_TARGET) $(DEBUG_TARGET) $(P100_TARGET) $(PTXAS_P100_TARGET) $(DEBUG_P100_TARGET)
