// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/Triggers/OrbitRadius.hpp"

#include <array>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace Triggers {

OrbitRadius::OrbitRadius(const std::vector<double>& radii) : radii_(radii) {}

bool OrbitRadius::operator()(
    const std::array<tnsr::I<double, 3, Frame::Inertial>, 2>&
        position_and_velocity,
    const TimeDelta& time_step) const {
  const auto& position = position_and_velocity[0];
  const auto& velocity = position_and_velocity[1];
  const double current_radius = get(magnitude(position));
  const double radial_velocity = (get<0>(position) * get<0>(velocity) +
                                  get<1>(position) * get<1>(velocity) +
                                  get<2>(position) * get<2>(velocity)) /
                                 current_radius;
  // factor 1.2 is for safety because the approximation is just linear
  // approximation and triggering it multiple times is not a problem
  const double next_radius =
      current_radius + 1.2 * radial_velocity * time_step.value();
  // NOLINTNEXTLINE(readability-use-anyofallof)
  for (const double radius : radii_) {
    if ((current_radius - radius) * (next_radius - radius) < 0.) {
      return true;
    }
  }
  return false;
}

// NOLINTNEXTLINE(google-runtime-references)
void OrbitRadius::pup(PUP::er& p) { p | radii_; }

PUP::able::PUP_ID OrbitRadius::my_PUP_ID = 0;  // NOLINT
}  // namespace Triggers
