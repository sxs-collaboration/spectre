// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace intrp::py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_cubic_spline(pybind11::module& m);
}  // namespace intrp::py_bindings
