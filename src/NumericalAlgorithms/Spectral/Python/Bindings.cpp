// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace Spectral::py_bindings {
void bind_basis(py::module& m);                  // NOLINT
void bind_quadrature(py::module& m);             // NOLINT
void bind_nodal_to_modal_matrix(py::module& m);  // NOLINT
void bind_modal_to_nodal_matrix(py::module& m);  // NOLINT
void bind_collocation_points(py::module& m);     // NOLINT
}  // namespace Spectral::py_bindings

namespace py_bindings {
void bind_mesh(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PySpectral, m) {  // NOLINT
  Spectral::py_bindings::bind_basis(m);
  Spectral::py_bindings::bind_quadrature(m);
  Spectral::py_bindings::bind_nodal_to_modal_matrix(m);
  Spectral::py_bindings::bind_modal_to_nodal_matrix(m);
  Spectral::py_bindings::bind_collocation_points(m);
  py_bindings::bind_mesh(m);
}
