// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_goldberg_to_nodal(pybind11::module& m);
void bind_nodal_to_goldberg(pybind11::module& m);
}  // namespace py_bindings
