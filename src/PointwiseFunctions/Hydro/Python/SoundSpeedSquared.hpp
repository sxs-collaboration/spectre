// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
template <typename DataType, size_t ThermodynamicDim>
void bind_soundSpeed(pybind11::module& m);
}  // namespace py_bindings
