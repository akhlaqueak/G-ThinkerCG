NVCC := nvcc
TARGET := run
DEBUG_TARGET := run_debug
P100_TARGET := run_p100
SRC := main.cu

INC := -I. -I.. -I../third_party/MPMCQueue/include -I../third_party/bliss-0.73 -I../system -I../common
CFLAGS := -std=c++20 -O3 -Xcompiler -fopenmp -lgomp -lpthread -DOPENMP
DEBUGFLAGS := -g -G -Xcompiler -gdwarf-4
REGFLAGS := --maxrregcount=64

BLISS_DIR := ../third_party/bliss-0.73
BLISS_LIB := -L$(BLISS_DIR)/bliss -lbliss

.PHONY: all debug p100 clean

all:
	$(NVCC) -arch=sm_80 $(REGFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -o $(TARGET)

debug:
	$(NVCC) -arch=sm_80 $(REGFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) $(DEBUGFLAGS) -o $(DEBUG_TARGET)

p100:
	$(NVCC) -arch=sm_60 $(REGFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -o $(P100_TARGET)

clean:
	rm -f $(TARGET) $(DEBUG_TARGET) $(P100_TARGET)
