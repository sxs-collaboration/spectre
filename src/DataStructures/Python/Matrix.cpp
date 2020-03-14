// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/Matrix.hpp"
#include "PythonBindings/BoundChecks.hpp"
#include "Utilities/MakeString.hpp"

namespace py = pybind11;

namespace py_bindings {

void bind_matrix(py::module& m) {  // NOLINT
  // Wrapper for basic Matrix operations
  py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
      .def(py::init<size_t, size_t>(), py::arg("rows"), py::arg("columns"))
      .def(py::init([](py::buffer buffer) {
             py::buffer_info info = buffer.request();
             // Sanity-check the buffer
             if (info.format != py::format_descriptor<double>::format()) {
               throw std::runtime_error(
                   "Incompatible format: expected a double array.");
             }
             if (info.ndim != 2) {
               throw std::runtime_error("Incompatible dimension.");
             }
             const auto rows = static_cast<size_t>(info.shape[0]);
             const auto columns = static_cast<size_t>(info.shape[1]);
             auto data = static_cast<double*>(info.ptr);
             return Matrix(rows, columns, data);
           }),
           py::arg("buffer"))
      // Expose the data as a Python buffer so it can be cast into Numpy arrays
      .def_buffer([](Matrix& matrix) {
        return py::buffer_info(
            matrix.data(),
            // Size of one scalar
            sizeof(double), py::format_descriptor<double>::format(),
            // Number of dimensions
            2,
            // Size of the buffer
            {matrix.rows(), matrix.columns()},
            // Stride for each index (in bytes). Data is stored
            // in column-major layout (see `Matrix.hpp`).
            {sizeof(double), sizeof(double) * matrix.spacing()});
      })
      .def_property_readonly(
          "shape",
          +[](const Matrix& self) {
            return std::tuple<size_t, size_t>(self.rows(), self.columns());
          })
      // __getitem__ and __setitem__ are the subscript operators (M[*,*]).
      .def(
          "__getitem__",
          +[](const Matrix& self, const std::tuple<size_t, size_t>& x) {
            matrix_bounds_check(self, std::get<0>(x), std::get<1>(x));
            return self(std::get<0>(x), std::get<1>(x));
          })
      // Need __str__ for converting to string/printing
      .def(
          "__str__",
          +[](const Matrix& self) { return std::string(MakeString{} << self); })
      .def(
          "__setitem__", +[](Matrix& self, const std::tuple<size_t, size_t>& x,
                             const double val) {
            matrix_bounds_check(self, std::get<0>(x), std::get<1>(x));
            self(std::get<0>(x), std::get<1>(x)) = val;
          });
}

}  // namespace py_bindings
