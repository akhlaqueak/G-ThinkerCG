# Compiler
NVCC = nvcc

# Target executable
TARGET = run

# Source file
SRC = main.cu

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)