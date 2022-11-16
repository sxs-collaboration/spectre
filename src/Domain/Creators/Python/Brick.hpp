// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace domain::creators::py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_brick(pybind11::module& m);
}  // namespace domain::creators::py_bindings
