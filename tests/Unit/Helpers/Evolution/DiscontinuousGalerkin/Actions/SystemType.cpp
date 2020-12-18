// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/SystemType.hpp"

#include <ostream>
#include <string>

namespace TestHelpers::evolution::dg::Actions {
std::ostream& operator<<(std::ostream& os, const SystemType t) noexcept {
  return os << (t == SystemType::Conservative ? std::string{"Conservative"}
                : t == SystemType::Nonconservative
                    ? std::string{"Nonconservative"}
                    : std::string{"Mixed"});
}
}  // namespace TestHelpers::evolution::dg::Actions
