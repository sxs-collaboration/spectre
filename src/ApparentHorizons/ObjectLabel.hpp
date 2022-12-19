// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>
#include <string>

namespace ah {
/// Labels for the objects in a binary system.
enum class ObjectLabel {
  /// The object along the positive x-axis in the grid frame
  A,
  /// The object along the negative x-axis in the grid frame
  B
};

std::string name(const ObjectLabel x);

std::ostream& operator<<(std::ostream& s, const ObjectLabel x);
}  // namespace ah
