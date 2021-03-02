// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"

namespace evolution::dg::subcell::Tags {
/// A record of which grid the TCI requested we use.
///
/// This is necessary because when using linear multistep methods for the time
/// integration we need to wait until the entire history is smooth before
/// returning to DG. For an Nth order integration in time, this means we need N
/// steps where the TCI has decided the solution is representable using DG.
///
/// The front of the history is the most recent entry.
struct TciGridHistory : db::SimpleTag {
  using type = std::deque<evolution::dg::subcell::ActiveGrid>;
};
}  // namespace evolution::dg::subcell::Tags
