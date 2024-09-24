// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Python/CombineH5.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "IO/H5/CombineH5.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_h5combine(py::module& m) {
  // Wrapper for combining h5 files
  m.def("combine_h5", &h5::combine_h5, py::arg("file_names"),
        py::arg("subfile_name"), py::arg("output"), py::arg("start-time"),
        py::arg("stop-time"), py::arg("check_src"));
}
}  // namespace py_bindings
