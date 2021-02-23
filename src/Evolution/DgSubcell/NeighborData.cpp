// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/NeighborData.hpp"

#include <ostream>
#include <pup.h>
#include <pup_stl.h>

#include "Utilities/StdHelpers.hpp"

namespace evolution::dg::subcell {
void pup(PUP::er& p, NeighborData& nhbr_data) noexcept {  // NOLINT
  p | nhbr_data.data_for_reconstruction;
  p | nhbr_data.max_variables_values;
  p | nhbr_data.min_variables_values;
}

void operator|(PUP::er& p, NeighborData& nhbr_data) noexcept {  // NOLINT
  pup(p, nhbr_data);
}

bool operator==(const NeighborData& lhs, const NeighborData& rhs) noexcept {
  return lhs.data_for_reconstruction == rhs.data_for_reconstruction and
         lhs.max_variables_values == rhs.max_variables_values and
         lhs.min_variables_values == rhs.min_variables_values;
}

bool operator!=(const NeighborData& lhs, const NeighborData& rhs) noexcept {
  return not(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const NeighborData& t) noexcept {
  using ::operator<<;
  return os << t.data_for_reconstruction << '\n'
            << t.max_variables_values << '\n'
            << t.min_variables_values;
}
}  // namespace evolution::dg::subcell
