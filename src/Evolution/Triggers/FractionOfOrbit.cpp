// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Triggers/FractionOfOrbit.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"

namespace Triggers {
FractionOfOrbit::FractionOfOrbit(const double fraction)
    : fraction_of_orbit_(fraction) {}

bool FractionOfOrbit::operator()(
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  for (auto i = functions_of_time.begin(); i != functions_of_time.end(); i++) {
    const auto* const rot_f_of_t = dynamic_cast<
        const domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
        (i->second.get()));
    if (rot_f_of_t != nullptr) {
      const double orbits_since_last_trigger =
          (rot_f_of_t->full_angle(time) -
           rot_f_of_t->full_angle(last_trigger_time_)) /
          (2.0 * M_PI);
      if (orbits_since_last_trigger >= fraction_of_orbit_) {
        last_trigger_time_ = time;
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

void FractionOfOrbit::pup(PUP::er& p) { p | fraction_of_orbit_; }

PUP::able::PUP_ID FractionOfOrbit::my_PUP_ID = 0;  // NOLINT
}  // namespace Triggers
