// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/Triggers/InsideHorizon.hpp"

#include <array>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/RadiusFunctions.hpp"

namespace Triggers {

bool InsideHorizon::operator()(
    const std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>&
        position_and_velocity,
    const std::array<double, 4>& worldtube_radius_params) const {
  const double orbit_radius = get(magnitude(position_and_velocity[0]));
  // optimization that will almost always be true
  if (LIKELY(orbit_radius > 1.99)) {
    return false;
  }
  const double wt_radius_inertial =
      CurvedScalarWave::Worldtube::smooth_broken_power_law(
          orbit_radius, worldtube_radius_params[0], worldtube_radius_params[1],
          worldtube_radius_params[2], worldtube_radius_params[3]);
  return orbit_radius + wt_radius_inertial < 1.99;
}

#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID InsideHorizon::my_PUP_ID = 0;  // NOLINT
#endif                                           // SPECTRE_USE_CHARM
}  // namespace Triggers
