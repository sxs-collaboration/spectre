// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "IO/H5/Dat.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_h5dat(py::module& m) {  // NOLINT
  // Wrapper for basic H5Dat operations
  py::class_<h5::Dat>(m, "H5Dat")
      .def("append",
           static_cast<void (h5::Dat::*)(const std::vector<double>&)>(
               &h5::Dat::append),
           py::arg("data"))
      .def("append",
           static_cast<void (h5::Dat::*)(const Matrix&)>(&h5::Dat::append),
           py::arg("data"))
      .def("append",
           static_cast<void (h5::Dat::*)(
               const std::vector<std::vector<double>>&)>(&h5::Dat::append),
           py::arg("data"))
      .def("get_legend", &h5::Dat::get_legend)
      .def("get_data", &h5::Dat::get_data)
      .def("get_data_subset", &h5::Dat::get_data_subset, py::arg("columns"),
           py::arg("first_row") = 0, py::arg("num_rows") = 1)
      .def("get_dimensions", &h5::Dat::get_dimensions)
      .def("get_header", &h5::Dat::get_header)
      .def("get_version", &h5::Dat::get_version);
}
}  // namespace py_bindings
