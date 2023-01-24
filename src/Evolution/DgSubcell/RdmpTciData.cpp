// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/RdmpTciData.hpp"

#include <ostream>
#include <pup.h>
#include <pup_stl.h>

namespace evolution::dg::subcell {
void pup(PUP::er& p, RdmpTciData& rdmp_tci_data) {  // NOLINT
  p | rdmp_tci_data.max_variables_values;
  p | rdmp_tci_data.min_variables_values;
}

void operator|(PUP::er& p, RdmpTciData& rdmp_tci_data) {  // NOLINT
  pup(p, rdmp_tci_data);
}

bool operator==(const RdmpTciData& lhs, const RdmpTciData& rhs) {
  return lhs.max_variables_values == rhs.max_variables_values and
         lhs.min_variables_values == rhs.min_variables_values;
}

bool operator!=(const RdmpTciData& lhs, const RdmpTciData& rhs) {
  return not(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const RdmpTciData& t) {
  using ::operator<<;
  return os << t.max_variables_values << '\n' << t.min_variables_values;
}
}  // namespace evolution::dg::subcell
