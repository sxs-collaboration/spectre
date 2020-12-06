// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Options/Auto.hpp"

#include <ostream>
#include <string>

#include "ErrorHandling/Error.hpp"

namespace Options {

std::ostream& operator<<(std::ostream& os, const AutoLabel label) noexcept {
  switch (label) {
    case AutoLabel::Auto:
      return os << "Auto";
    case AutoLabel::None:
      return os << "None";
    default:
      ERROR("Invalid label");
  }
}

}  // namespace Options
