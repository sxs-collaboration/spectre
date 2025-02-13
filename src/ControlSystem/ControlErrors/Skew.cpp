// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Skew.hpp"

#include <optional>
#include <pup.h>

namespace control_system::ControlErrors {
Skew::Skew(const Skew& rhs) : suggested_timescale_(rhs.suggested_timescale_) {}

Skew& Skew::operator=(const Skew& rhs) {
  suggested_timescale_ = rhs.get_suggested_timescale();
  return *this;
}

std::optional<double> Skew::get_suggested_timescale() const {
  return suggested_timescale_;
}

void Skew::reset() { suggested_timescale_ = std::nullopt; }

void Skew::pup(PUP::er& p) {
  // No need to pup the DataBoxes or the inclination angles. Their data is
  // always temporary
  p | suggested_timescale_;
}
}  // namespace control_system::ControlErrors
