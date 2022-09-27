// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <limits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"

namespace TestHelpers::tenex {
// Get a placeholder value for a Tensor component based on the data type of its
// components
template <typename DataType>
struct component_placeholder_value;

template <>
struct component_placeholder_value<double> {
  static constexpr double value = std::numeric_limits<double>::max();
};

template <>
struct component_placeholder_value<std::complex<double>> {
  static constexpr std::complex<double> value = std::complex<double>(
      std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
};

template <>
struct component_placeholder_value<DataVector> {
  static constexpr double value = component_placeholder_value<double>::value;
};

template <>
struct component_placeholder_value<ComplexDataVector> {
  static constexpr std::complex<double> value =
      component_placeholder_value<std::complex<double>>::value;
};
}  // namespace TestHelpers::tenex
