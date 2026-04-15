# Compiler
NVCC = nvcc

# Target executable
TARGET = run

# Source file
SRC = main.cu

# Default flags (normal build)
NVFLAGS = -DDEBUG_TOGGLE=0 -O3

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVFLAGS) $(SRC) -o $(TARGET)

# Debug build
debug:
	$(NVCC) -DDEBUG_TOGGLE=1 -g -G $(SRC) -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)