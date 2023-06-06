// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Options/Context.hpp"

#include <ostream>

namespace Options {
std::ostream& operator<<(std::ostream& s, const Context& c) {
  s << c.context;
  if (c.line >= 0 and c.column >= 0) {
    s << "At line " << c.line + 1 << " column " << c.column + 1 << ":\n";
  }
  return s;
}
}  // namespace Options
