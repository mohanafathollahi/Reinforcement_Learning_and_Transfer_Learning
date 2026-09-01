#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef __cplusplus
extern "C"{
#endif
// Declare the mangled name of the function
extern void runSimulation_r2(const double data_inputs[7], double data_outputs[20]);
}
namespace py = pybind11;

void run_simulation(py::array_t<double> input, py::array_t<double> output) {
    // Check input and output array sizes
    if (input.size() != 7) {
        throw std::runtime_error("Input array must have 7 elements.");
    }
    if (output.size() != 20) {
        throw std::runtime_error("Output array must have 20 elements.");
    }

    // Request buffers for the numpy arrays
    py::buffer_info input_buf = input.request();
    py::buffer_info output_buf = output.request();

    // Convert to C++ pointer arrays
    double* input_ptr = static_cast<double*>(input_buf.ptr);
    double* output_ptr = static_cast<double*>(output_buf.ptr);

    // Call the mangled C++ function
    runSimulation_r2(input_ptr, output_ptr);
}

PYBIND11_MODULE(simulation, m) {
    m.def("run_simulation", &run_simulation, "Run the simulation with given inputs and outputs",
          py::arg("input"), py::arg("output"));
}
