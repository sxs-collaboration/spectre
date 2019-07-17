// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/python.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "Informer/InfoFromBuild.hpp"
#include "Utilities/GetOutput.hpp"
namespace bp = boost::python;

namespace py_bindings {
void bind_info_at_compile() {
  // Wrapper to make Build Info available from python
  bp::def("spectre_version", &spectre_version);
  bp::def("spectre_major_version", &spectre_major_version);
  bp::def("spectre_minor_version", &spectre_minor_version);
  bp::def("spectre_patch_version", &spectre_patch_version);
  bp::def("unit_test_path", &unit_test_path);
}
}  // namespace py_bindings
