// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace PowerMonitors::py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_power_monitors(pybind11::module& m);
}  // namespace PowerMonitors::py_bindings
