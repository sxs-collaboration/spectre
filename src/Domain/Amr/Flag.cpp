// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Amr/Flag.hpp"
#include "ErrorHandling/Error.hpp"

#include <ostream>

namespace amr {

std::ostream& operator<<(std::ostream& os, const Flag& flag) {
  switch (flag) {
    case Flag::Undefined:
      os << "Undefined";
      break;
    case Flag::Join:
      os << "Join";
      break;
    case Flag::DecreaseResolution:
      os << "DecreaseResolution";
      break;
    case Flag::DoNothing:
      os << "DoNothing";
      break;
    case Flag::IncreaseResolution:
      os << "IncreaseResolution";
      break;
    case Flag::Split:
      os << "Split";
      break;
    default:
      ERROR("An undefined flag was passed to the stream operator.");
  }
  return os;
}
}  // namespace amr
