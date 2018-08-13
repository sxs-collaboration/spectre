// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace grmhd {
namespace ValenciaDivClean {

std::array<DataVector, 9> characteristic_speeds(
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& alfven_speed_squared,
    const tnsr::i<DataVector, 3>& normal) noexcept {
  auto characteristic_speeds =
      make_array<9, DataVector>(-1.0 * get(dot_product(normal, shift)));

  const auto auxiliary_speeds =
      RelativisticEuler::Valencia::characteristic_speeds(
          lapse, shift, spatial_velocity, spatial_velocity_squared,
          Scalar<DataVector>{get(sound_speed_squared) +
                             get(alfven_speed_squared) *
                                 (1.0 - get(sound_speed_squared))},
          normal);

  characteristic_speeds[0] -= get(lapse);
  characteristic_speeds[1] = auxiliary_speeds[0];
  for (size_t i = 2; i < 7; ++i) {
    // auxiliary_speeds[1], auxiliary_speeds[2], and auxiliary_speeds[3]
    // are the same.
    gsl::at(characteristic_speeds, i) = auxiliary_speeds[2];
  }
  characteristic_speeds[7] = auxiliary_speeds[4];
  characteristic_speeds[8] += get(lapse);

  return characteristic_speeds;
}

}  // namespace ValenciaDivClean
}  // namespace grmhd
/// \endcond
