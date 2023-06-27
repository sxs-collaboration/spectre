// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/LowerSpatialFourVelocity.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"

namespace hydro::Tags {
void LowerSpatialFourVelocityCompute::function(
    const gsl::not_null<tnsr::i<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const Scalar<DataVector>& lorentz_factor) {
  raise_or_lower_index(result, spatial_velocity, spatial_metric);
  for (size_t d = 0; d < 3; ++d) {
    result->get(d) *= get(lorentz_factor);
  }
}

}  // namespace hydro::Tags
