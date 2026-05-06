CXX = g++
CXXFLAGS = -O3
TARGET = run
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(SRC) -o $(TARGET) $(CXXFLAGS)

clean:
	rm -f $(TARGET)