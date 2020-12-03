// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

namespace TestHelpers::evolution::dg::Actions {
enum SystemType { Conservative, Nonconservative, Mixed };

std::ostream& operator<<(std::ostream& os, SystemType t) noexcept;
}  // namespace TestHelpers::evolution::dg::Actions
