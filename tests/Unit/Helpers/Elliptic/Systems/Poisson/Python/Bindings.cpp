// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace TestHelpers {
namespace Poisson {

namespace py_bindings {
void bind_dg_schemes(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyPoissonTestHelpers, m) {  // NOLINT
  py_bindings::bind_dg_schemes(m);
}

}  // namespace Poisson
}  // namespace TestHelpers
