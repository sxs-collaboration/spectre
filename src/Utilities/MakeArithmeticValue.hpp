// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines make_arithmetic_value

#pragma once

#include "DataStructures/DataVector.hpp"
#include "Utilities/ForceInline.hpp"

/*!
 *  \ingroup Utilities
 *  \brief Returns a DataVector the same size as `input`, with each element
 * equal to `value`.
 */
SPECTRE_ALWAYS_INLINE DataVector make_arithmetic_value(const DataVector& input,
                                                       double value) {
  return DataVector(input.size(), value);
}

/*!
 *  \ingroup Utilities
 *  \brief Returns the double `value`.
 *
 *  \details The argument `input` is unused, but is present so that
 * `make_arithmetic_value` has the same interface for doubles and DataVectors.
 */
SPECTRE_ALWAYS_INLINE double make_arithmetic_value(double /*input*/,
                                                   double value) {
  return value;
}
