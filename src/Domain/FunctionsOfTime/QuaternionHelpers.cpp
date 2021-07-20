// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/QuaternionHelpers.hpp"

#include <boost/math/quaternion.hpp>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

DataVector quaternion_to_datavector(
    const boost::math::quaternion<double>& input) noexcept {
  return DataVector{input.R_component_1(), input.R_component_2(),
                    input.R_component_3(), input.R_component_4()};
}

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

// Normalize a `boost::math::quaternion`
void normalize_quaternion(
    const gsl::not_null<boost::math::quaternion<double>*> input) noexcept {
  *input /= abs(*input);
}
