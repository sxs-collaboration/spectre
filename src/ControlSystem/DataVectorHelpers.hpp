// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"

/// \ingroup DataStructuresGroup
/// Convert a std::array to a DataVector. Currently only instantiated for N=3.
template <size_t N>
DataVector array_to_datavector(const std::array<double, N>& arr);
