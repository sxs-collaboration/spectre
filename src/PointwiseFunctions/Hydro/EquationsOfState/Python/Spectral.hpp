// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace EquationsOfState::py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_spectral(pybind11::module& m);
}  // namespace EquationsOfState::py_bindings
