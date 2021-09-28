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
/// Holds neighbor data needed for the TCI and reconstruction.
///
/// The `max_variables_values` and `min_variables_values` are used for the
/// relaxed discrete maximum principle troubled-cell indicator.
struct NeighborData {
  std::vector<double> data_for_reconstruction{};
  std::vector<double> max_variables_values{};
  std::vector<double> min_variables_values{};
};

void pup(PUP::er& p, NeighborData& nhbr_data);  // NOLINT

void operator|(PUP::er& p, NeighborData& nhbr_data);  // NOLINT

bool operator==(const NeighborData& lhs, const NeighborData& rhs);

bool operator!=(const NeighborData& lhs, const NeighborData& rhs);

std::ostream& operator<<(std::ostream& os, const NeighborData& t);
}  // namespace evolution::dg::subcell
