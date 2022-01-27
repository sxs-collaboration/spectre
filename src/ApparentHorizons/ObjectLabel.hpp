// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>
#include <string>

namespace ah {
/// Labels for the objects in a binary system.
enum class ObjectLabel { A, B };

std::string name(const ObjectLabel x);

std::ostream& operator<<(std::ostream& s, const ObjectLabel x);
}  // namespace ah
