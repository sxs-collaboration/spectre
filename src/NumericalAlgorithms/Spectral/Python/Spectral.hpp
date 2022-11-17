// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace Spectral::py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_basis(pybind11::module& m);
void bind_quadrature(pybind11::module& m);
void bind_modal_to_nodal_matrix(pybind11::module& m);
void bind_nodal_to_modal_matrix(pybind11::module& m);
void bind_collocation_points(pybind11::module& m);
}  // namespace Spectral::py_bindings
