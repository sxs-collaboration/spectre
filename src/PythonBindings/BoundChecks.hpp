// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include "Utilities/PrettyType.hpp"

namespace py_bindings {
/*!
 * \ingroup PythonBindingsGroup
 * \brief Check if a vector-like object access is in bounds. Throws
 * std::runtime_error if it is not.
 */
template <typename T>
void bounds_check(const T& t, const size_t i) {
  if (i >= t.size()) {
    throw std::runtime_error{"Out of bounds access (" + std::to_string(i) +
                             ") into " + pretty_type::short_name<T>() +
                             " of size " + std::to_string(t.size())};
  }
}
/*!
 * \ingroup PythonBindingsGroup
 * \brief Check if a matrix-like object access is in bounds. Throws
 * std::runtime_error if it is not.
 */
template <typename T>
void matrix_bounds_check(const T& matrix, const size_t row,
                         const size_t column) {
  if (row >= matrix.rows() or column >= matrix.columns()) {
    throw std::runtime_error{"Out of bounds access (" + std::to_string(row) +
                             ", " + std::to_string(column) +
                             ") into Matrix of size (" +
                             std::to_string(matrix.rows()) + ", " +
                             std::to_string(matrix.columns()) + ")"};
  }
}
}  // namespace py_bindings
