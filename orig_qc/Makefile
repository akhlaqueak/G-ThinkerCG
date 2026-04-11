NVCC ?= nvcc
NVCCFLAGS ?=

TARGET := run
SRC := main.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
