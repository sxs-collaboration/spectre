// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Triggers/FractionOfOrbit.hpp"
#include <iostream>

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>

namespace DenseTriggers {
FractionOfOrbit::FractionOfOrbit(const double fraction,
                                 const double initial_time)
    : fraction_of_orbit_(fraction), last_trigger_time_(initial_time) {}

void FractionOfOrbit::pup(PUP::er& p) {
  DenseTrigger::pup(p);
  p | fraction_of_orbit_;
  p | last_trigger_time_;
}

std::optional<bool> FractionOfOrbit::is_triggered_impl(
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  for (auto i = functions_of_time.begin(); i != functions_of_time.end(); i++) {
    const auto* const rot_f_of_t = dynamic_cast<
        const domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
        (i->second.get()));
    if (rot_f_of_t != nullptr) {
      if (last_trigger_time_ > time) {
        return false;
      }
      const double orbits_since_last_trigger =
          abs((rot_f_of_t->full_angle(time) -
               rot_f_of_t->full_angle(last_trigger_time_))) /
          (2.0 * M_PI);
      if (orbits_since_last_trigger >= fraction_of_orbit_) {
        last_trigger_time_ = time;
        std::cout << "time when triggered: " << time;
        return true;
      } else {
        return false;
      }
    }
  }
  ERROR(
      "FractionOfOrbit trigger can only be used when the rotation map is "
      "active");
}

PUP::able::PUP_ID FractionOfOrbit::my_PUP_ID = 0;  // NOLINT
}  // namespace DenseTriggers
