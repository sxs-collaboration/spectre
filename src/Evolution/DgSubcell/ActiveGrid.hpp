// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

namespace evolution::dg::subcell {
/// \ingroup DgSubcellGroup
/// The grid that is currently being used for the DG-subcell evolution.
enum class ActiveGrid { Dg, Subcell };

std::ostream& operator<<(std::ostream& os, ActiveGrid active_grid) noexcept;
}  // namespace evolution::dg::subcell
