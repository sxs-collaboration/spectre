// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace RelativisticEuler::Solutions::py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_tov(pybind11::module& m);
}  // namespace RelativisticEuler::Solutions::py_bindings
