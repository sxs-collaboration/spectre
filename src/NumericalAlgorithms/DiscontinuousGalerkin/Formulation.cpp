// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"

#include <ostream>

#include "ErrorHandling/Error.hpp"

namespace dg {
std::ostream& operator<<(std::ostream& os, const Formulation t) noexcept {
  switch (t) {
    case Formulation::StrongInertial:
      return os << "StrongInertial";
    case Formulation::WeakInertial:
      return os << "WeakInertial";
    default:
      ERROR("Unknown DG formulation.");
  }
}
}  // namespace dg
