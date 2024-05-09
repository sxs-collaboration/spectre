// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Python/Cce.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "IO/H5/Cce.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_h5cce(py::module& m) {
  // Wrapper for basic H5Cce operations
  py::class_<h5::Cce>(m, "H5Cce")
      .def("append", &h5::Cce::append, py::arg("data"))
      .def("get_legend", &h5::Cce::get_legend)
      .def("get_data", py::overload_cast<>(&h5::Cce::get_data, py::const_))
      .def(
          "get_data",
          py::overload_cast<const std::string&>(&h5::Cce::get_data, py::const_),
          py::arg("bondi_variable_name"))
      .def("get_data_subset",
           py::overload_cast<const std::vector<size_t>&, size_t, size_t>(
               &h5::Cce::get_data_subset, py::const_),
           py::arg("ell"), py::arg("first_row") = 0, py::arg("num_rows") = 1)
      .def("get_data_subset",
           py::overload_cast<const std::string&, const std::vector<size_t>&,
                             size_t, size_t>(&h5::Cce::get_data_subset,
                                             py::const_),
           py::arg("bondi_variable_name"), py::arg("ell"),
           py::arg("first_row") = 0, py::arg("num_rows") = 1)
      .def("get_dimensions", &h5::Cce::get_dimensions)
      .def("get_header", &h5::Cce::get_header)
      .def("get_version", &h5::Cce::get_version);
}
}  // namespace py_bindings
