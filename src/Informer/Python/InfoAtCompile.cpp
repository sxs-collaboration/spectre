// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <string>

#include "Informer/InfoFromBuild.hpp"

namespace py = pybind11;

namespace py_bindings {
void bind_info_at_compile(py::module& m) {  // NOLINT
  // Wrapper to make Build Info available from python
  m.def("spectre_version", &spectre_version);
  m.def("unit_test_src_path", &unit_test_src_path);
  m.def("unit_test_build_path", &unit_test_build_path);
}
}  // namespace py_bindings
