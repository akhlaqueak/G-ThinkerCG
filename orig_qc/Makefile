NVCC ?= nvcc
NVCCFLAGS ?=

TARGET := cuQC
SRC := main.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
