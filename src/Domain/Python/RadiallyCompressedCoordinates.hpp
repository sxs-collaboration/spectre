// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace domain::py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_radially_compressed_coordinates(pybind11::module& m);
}  // namespace domain::py_bindings
