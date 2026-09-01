# Set your pybind11 include path and Python config
PYBIND11_DIR = $(shell python3.11 -m pybind11 --includes)
PYTHON_INCLUDE = $(shell python3.11 -m pybind11 --includes)

# Compiler and linker flags
CXX = g++
CXXFLAGS = -O3 -Wall -shared  -std=c++11 -fpic

# Static library location (replace with actual path)
LIB_PATH = ./runSimulation_r2.a

# Target extension module
TARGET = simulation$(shell python3.11-config --extension-suffix)

all: $(TARGET)

$(TARGET): wrapper.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_PATH) $(PYTHON_INCLUDE)

wrapper.o: wrapper.cpp
	$(CXX) $(CXXFLAGS) -c wrapper.cpp $(PYTHON_INCLUDE)

clean:
	rm -f *.o $(TARGET)


