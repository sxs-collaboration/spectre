// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace gr {
namespace Solutions {
namespace py_bindings {
void bind_tov(py::module& m);  // NOLINT
}  // namespace py_bindings
}  // namespace Solutions
}  // namespace gr

PYBIND11_MODULE(_PyGeneralRelativitySolutions, m) {  // NOLINT
  gr::Solutions::py_bindings::bind_tov(m);
}
