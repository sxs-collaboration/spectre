// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Python/CombineH5.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "IO/H5/CombineH5.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_h5combine(py::module& m) {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  // Wrapper for combining h5 files
  m.def("combine_h5", &h5::combine_h5, py::arg("file_names"),
        py::arg("subfile_name"), py::arg("output"), py::arg("start-time"),
        py::arg("stop-time"), py::arg("blocks_to_combine"),
        py::arg("check_src"));
}
}  // namespace py_bindings
