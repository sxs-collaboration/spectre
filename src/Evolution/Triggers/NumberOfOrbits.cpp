// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Triggers/NumberOfOrbits.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"

namespace Triggers {
NumberOfOrbits::NumberOfOrbits(const double orbits)
    : number_of_orbits_(orbits) {}

bool NumberOfOrbits::operator()(
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  for (auto i = functions_of_time.begin(); i != functions_of_time.end(); i++) {
    const auto* const rot_f_of_t = dynamic_cast<
        const domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
        (i->second.get()));
    if (rot_f_of_t != nullptr) {
      const double calculated_orbits =
          rot_f_of_t->full_angle(time) / (2.0 * M_PI);
      return calculated_orbits >= number_of_orbits_;
    }
  }
  ERROR(
      "NumberOfOrbits trigger can only be used when the rotation map is "
      "active");
}

void NumberOfOrbits::pup(PUP::er& p) { p | number_of_orbits_; }

PUP::able::PUP_ID NumberOfOrbits::my_PUP_ID = 0;  // NOLINT
}  // namespace Triggers
