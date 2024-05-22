// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Python/ModalVector.hpp"

#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <utility>

#include "DataStructures/ModalVector.hpp"
#include "PythonBindings/BoundChecks.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_modalvector(py::module& m) {
  // Wrapper for basic ModalVector operations
  py::class_<ModalVector>(m, "ModalVector", py::buffer_protocol())
      .def(py::init<size_t>(), py::arg("size"))
      .def(py::init<size_t, double>(), py::arg("size"), py::arg("fill"))
      .def(py::init([](const std::vector<double>& values) {
             ModalVector result{values.size()};
             std::copy(values.begin(), values.end(), result.begin());
             return result;
           }),
           py::arg("values"))
      .def(py::init([](const py::buffer& buffer, const bool copy) {
             py::buffer_info info = buffer.request();
             // Sanity-check the buffer
             if (info.format != py::format_descriptor<double>::format()) {
               throw std::runtime_error(
                   "Incompatible format: expected a double array.");
             }
             if (info.ndim != 1) {
               throw std::runtime_error("Incompatible dimension.");
             }
             const auto size = static_cast<size_t>(info.shape[0]);
             auto data = static_cast<double*>(info.ptr);
             if (copy) {
               ModalVector result{size};
               std::copy_n(data, result.size(), result.begin());
               return result;
             } else {
               // Create a non-owning ModalVector from the buffer
               return ModalVector{data, size};
             }
           }),
           py::arg("buffer"), py::arg("copy") = true)
      // Expose the data as a Python buffer so it can be cast into Numpy arrays
      .def_buffer([](ModalVector& data_vector) {
        return py::buffer_info(data_vector.data(),
                               // Size of one scalar
                               sizeof(double),
                               py::format_descriptor<double>::format(),
                               // Number of dimensions
                               1,
                               // Size of the buffer
                               {data_vector.size()},
                               // Stride for each index (in bytes)
                               {sizeof(double)});
      })
      .def(
          "__iter__",
          [](const ModalVector& t) {
            return py::make_iterator(t.begin(), t.end());
          },
          // Keep object alive while iterator exists
          py::keep_alive<0, 1>())
      // __len__ is for being able to write len(my_data_vector) in python
      .def("__len__", [](const ModalVector& t) { return t.size(); })
      // __getitem__ and __setitem__ are the subscript operators (operator[]).
      // To define (and overload) operator() use __call__
      .def(
          "__getitem__",
          +[](const ModalVector& t, const size_t i) {
            bounds_check(t, i);
            return t[i];
          })
      .def(
          "__setitem__",
          +[](ModalVector& t, const size_t i, const double v) {
            bounds_check(t, i);
            t[i] = v;
          })
      // Need __str__ for converting to string/printing
      .def(
          "__str__", +[](const ModalVector& t) { return get_output(t); })
      // repr allows you to output the object in an interactive python terminal
      // using obj to get the "string REPResenting the object".
      .def(
          "__repr__", +[](const ModalVector& t) { return get_output(t); })
      .def(py::self += py::self)
      // Need to do math explicitly converting to ModalVector because we don't
      // want to represent all the possible expression template types
      .def(
          "abs", +[](const ModalVector& t) { return ModalVector{abs(t)}; })
      .def(
          "__mul__",
          +[](const ModalVector& self, const double other) {
            return ModalVector{self * other};
          })
      .def(
          "__rmul__",
          +[](const ModalVector& self, const double other) {
            return ModalVector{other * self};
          })
      .def(
          "__div__",
          +[](const ModalVector& self, const double other) {
            return ModalVector{self / other};
          })
      .def(
          "__truediv__",
          +[](const ModalVector& self, const double other) {
            return ModalVector{self / other};
          })

      // ModalVector-ModalVector math
      .def(
          "__add__",
          +[](const ModalVector& self, const ModalVector& other) {
            return ModalVector{self + other};
          })
      .def(
          "__sub__",
          +[](const ModalVector& self, const ModalVector& other) {
            return ModalVector{self - other};
          })

      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self == py::self)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self != py::self)
      .def(
          "__neg__", +[](const ModalVector& t) { return ModalVector{-t}; });
  py::implicitly_convertible<py::array, ModalVector>();
}
}  // namespace py_bindings
