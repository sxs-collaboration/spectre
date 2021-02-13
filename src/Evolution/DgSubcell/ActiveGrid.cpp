// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/ActiveGrid.hpp"

#include <ostream>

#include "Utilities/ErrorHandling/Error.hpp"

namespace evolution::dg::subcell {
std::ostream& operator<<(std::ostream& os, ActiveGrid active_grid) noexcept {
  switch (active_grid) {
    case ActiveGrid::Dg:
      return os << "Dg";
    case ActiveGrid::Subcell:
      return os << "Subcell";
    default:
      ERROR("ActiveGrid must be either 'Dg' or 'Subcell' but has value "
            << static_cast<int>(active_grid));
  }
}
}  // namespace evolution::dg::subcell
