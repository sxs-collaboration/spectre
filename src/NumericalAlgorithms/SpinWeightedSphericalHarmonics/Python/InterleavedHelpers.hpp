// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "DataStructures/DataVector.hpp"

namespace py_bindings::detail {
// The bindings pass spin-weighted complex data as real, interleaved
// [re, im, re, im, ...] arrays so they can be constructed directly from numpy.

template <typename ComplexVector>
ComplexVector interleaved_to_complex(const DataVector& interleaved) {
  if (interleaved.size() % 2 != 0) {
    throw std::invalid_argument(
        "Interleaved [re, im, ...] array must have an even number of "
        "entries, not " +
        std::to_string(interleaved.size()) + ".");
  }
  ComplexVector result{interleaved.size() / 2};
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] =
        std::complex<double>(interleaved[2 * i], interleaved[2 * i + 1]);
  }
  return result;
}

template <typename ComplexVector>
DataVector complex_to_interleaved(const ComplexVector& complex_values) {
  DataVector result{2 * complex_values.size()};
  for (size_t i = 0; i < complex_values.size(); ++i) {
    result[2 * i] = complex_values[i].real();
    result[2 * i + 1] = complex_values[i].imag();
  }
  return result;
}
}  // namespace py_bindings::detail
