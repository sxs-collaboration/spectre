// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace Spectral::py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_logical_coordinates(pybind11::module& m);
}  // namespace Spectral::py_bindings
