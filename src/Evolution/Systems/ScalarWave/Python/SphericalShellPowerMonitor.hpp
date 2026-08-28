// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace ScalarWave::power_monitor::py_bindings {

// NOLINTNEXTLINE(google-runtime-references)
void bind_spherical_shell_power_monitor(pybind11::module& m);

}  // namespace ScalarWave::power_monitor::py_bindings
