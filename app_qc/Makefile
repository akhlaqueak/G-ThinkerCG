NVCC := nvcc
CXX := g++
TARGET := run
SRC := main.cu
BIN_TO_SER := binToSer
BIN_TO_SER_SRC := binToSer.cpp

INC := -I.. -I../common/MPMCQueue/include -I../common/bliss -I../system/ -I../common
CFLAGS := -std=c++20 -O3 -Xcompiler -fopenmp -lgomp -lpthread -DOPENMP
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -pthread
DEBUGFLAGS := -g -G -Xcompiler -gdwarf-4
REGFLAGS := --maxrregcount=64

BLISS_DIR := ../common
BLISS_LIB := -L$(BLISS_DIR)/bliss -lbliss

ARCH_SM80 := -arch=sm_80
ARCH_SM60 := -arch=sm_60

.PHONY: all a100 debug_sm60 debug_sm80 tools clean

all: a100

tools: $(BIN_TO_SER)

$(BIN_TO_SER): $(BIN_TO_SER_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

a100:
	$(NVCC) $(ARCH_SM80) $(REGFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -o $(TARGET)

debug_p100:
	$(NVCC) $(ARCH_SM60) $(REGFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) $(DEBUGFLAGS) -o $(TARGET)

p100:
	$(NVCC) $(ARCH_SM60) $(REGFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) -o $(TARGET)

debug:
	$(NVCC) $(ARCH_SM80) $(REGFLAGS) $(CFLAGS) $(INC) $(BLISS_LIB) $(SRC) $(DEBUGFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET) $(BIN_TO_SER)
