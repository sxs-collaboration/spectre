// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helper functions for converting between `DataVector`s and boost
/// quaternions.

#pragma once

#include <boost/math/quaternion.hpp>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

/// Convert a `boost::math::quaternion` to a `DataVector`
DataVector quaternion_to_datavector(
    const boost::math::quaternion<double>& input) noexcept {
  return DataVector{input.R_component_1(), input.R_component_2(),
                    input.R_component_3(), input.R_component_4()};
}

/// \brief Convert a `DataVector` to a `boost::math::quaternion`
///
/// \details To convert to a quaternion, a `DataVector` must have either 3 or 4
/// components. If it has 3 components, the quaternion will be constructed with
/// 0 scalar part while the vector part is the `DataVector`. If the `DataVector`
/// has 4 components, the quaternion is just the `DataVector` itself.
boost::math::quaternion<double> datavector_to_quaternion(
    const DataVector& input) noexcept {
  ASSERT(input.size() == 3 or input.size() == 4,
         "To form a quaternion, a DataVector can either have 3 or 4 components "
         "only. This DataVector has "
             << input.size() << " components.");
  if (input.size() == 3) {
    return boost::math::quaternion<double>(0.0, input[0], input[1], input[2]);
  } else {
    return boost::math::quaternion<double>(input[0], input[1], input[2],
                                           input[3]);
  }
}
/// Normalize a `boost::math::quaternion`
template <typename T>
void normalize_quaternion(
    const gsl::not_null<boost::math::quaternion<T>*> input) noexcept {
  *input /= abs(*input);
}
