// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Characteristics.hpp"

#include <algorithm>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree::Tags {

void LargestCharacteristicSpeedCompute::function(
    const gsl::not_null<double*> speed, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3>& spatial_metric) {
  const auto shift_magnitude = magnitude(shift, spatial_metric);
  *speed = max(get(shift_magnitude) + get(lapse));
}

}  // namespace ForceFree::Tags
