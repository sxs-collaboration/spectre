// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ObjectLabel.hpp"

#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Literals.hpp"

namespace domain {
std::string name(const ObjectLabel x) {
  if (x == ObjectLabel::A) {
    return "A"s;
  } else if (x == ObjectLabel::B) {
    return "B"s;
  } else if (x == ObjectLabel::None) {
    return ""s;
  } else {
    ERROR("Unknown object label!");
  }
}

std::ostream& operator<<(std::ostream& s, const ObjectLabel x) {
  return s << name(x);
}
}  // namespace domain
