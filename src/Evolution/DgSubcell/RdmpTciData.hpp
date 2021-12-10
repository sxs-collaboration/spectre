// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace evolution::dg::subcell {
/// Holds data needed for the relaxed discrete maximum principle
/// troubled-cell indicator.
struct RdmpTciData {
  std::vector<double> max_variables_values{};
  std::vector<double> min_variables_values{};
};

void pup(PUP::er& p, RdmpTciData& rdmp_tci_data);  // NOLINT

void operator|(PUP::er& p, RdmpTciData& rdmp_tci_data);  // NOLINT

bool operator==(const RdmpTciData& lhs, const RdmpTciData& rhs);

bool operator!=(const RdmpTciData& lhs, const RdmpTciData& rhs);

std::ostream& operator<<(std::ostream& os, const RdmpTciData& t);
}  // namespace evolution::dg::subcell
