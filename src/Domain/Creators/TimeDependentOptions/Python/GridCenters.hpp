// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace domain::creators::time_dependent_options::py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_grid_centers(pybind11::module& m);
}  // namespace domain::creators::time_dependent_options::py_bindings
