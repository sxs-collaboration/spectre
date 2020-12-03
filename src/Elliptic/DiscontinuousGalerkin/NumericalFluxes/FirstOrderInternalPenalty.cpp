// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"

namespace elliptic::dg::NumericalFluxes {

DataVector penalty(const DataVector& element_size, const size_t num_points,
                   const double penalty_parameter) noexcept {
  return penalty_parameter * square(num_points) / element_size;
}

}  // namespace elliptic::dg::NumericalFluxes
