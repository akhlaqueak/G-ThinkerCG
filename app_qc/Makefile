NVCC := nvcc
TARGET := run
SRC := main.cu

INC := -I.. -I../common/MPMCQueue/include -I../common/bliss -I../system/ -I../common
CFLAGS := -std=c++20 -O3 -Xcompiler -fopenmp -lgomp -lpthread -DOPENMP
DEBUGFLAGS := -g -G -Xcompiler -gdwarf-4
REGFLAGS := --maxrregcount=64

BLISS_DIR := ../common
BLISS_LIB := -L$(BLISS_DIR)/bliss -lbliss

ARCH_SM80 := -arch=sm_80
ARCH_SM60 := -arch=sm_60

.PHONY: all a100 debug_sm60 debug_sm80

all: a100

a100:
	$(NVCC) $(ARCH_SM80) $(REGFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -o $(TARGET)

debug_p100:
	$(NVCC) $(ARCH_SM60) $(REGFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) $(DEBUGFLAGS) -o $(TARGET)

p100:
	$(NVCC) $(ARCH_SM60) $(REGFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -o $(TARGET)

debug_a100:
	$(NVCC) $(ARCH_SM80) $(REGFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) $(DEBUGFLAGS) -o $(TARGET)
