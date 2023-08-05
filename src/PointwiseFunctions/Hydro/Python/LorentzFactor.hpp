// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
void bind_lorentz(pybind11::module& m);

void bind_lorentz_factor(pybind11::module& m);
}  // namespace py_bindings
