// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingParameters.hpp"

#include <pup.h>

namespace ScalarTensor {

CouplingParameterOptions::CouplingParameterOptions(const double linear_in,
                                                   const double quadratic_in,
                                                   const double quartic_in)
    : linear(linear_in), quadratic(quadratic_in), quartic(quartic_in) {}

void CouplingParameterOptions::pup(PUP::er& p) {
  p | linear;
  p | quadratic;
  p | quartic;
}

bool operator==(const CouplingParameterOptions& lhs,
                const CouplingParameterOptions& rhs) {
  return lhs.linear == rhs.linear and lhs.quadratic == rhs.quadratic and
         lhs.quartic == rhs.quartic;
}

bool operator!=(const CouplingParameterOptions& lhs,
                const CouplingParameterOptions& rhs) {
  return not(lhs == rhs);
}

}  // namespace ScalarTensor
