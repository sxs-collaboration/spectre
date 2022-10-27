// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Characteristics.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree::Tags {

void LargestCharacteristicSpeedCompute::function(
    const gsl::not_null<double*> speed, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3>& shift,
    const tnsr::i<DataVector, 3>& unit_normal) {
  Scalar<DataVector> normal_dot_shift(get(lapse).size());
  dot_product(make_not_null(&normal_dot_shift), unit_normal, shift);

  *speed = fmax(max(abs(-get(normal_dot_shift) - get(lapse))),
                max(abs(-get(normal_dot_shift) + get(lapse))));
}

}  // namespace ForceFree::Tags
