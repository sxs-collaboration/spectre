// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/ObjectLabel.hpp"

#include <ostream>

#include "Utilities/Literals.hpp"

namespace ah {
std::string name(const ObjectLabel x) {
  return x == ObjectLabel::A ? "A"s : "B"s;
}

std::ostream& operator<<(std::ostream& s, const ObjectLabel x) {
  return s << name(x);
}
}  // namespace ah
