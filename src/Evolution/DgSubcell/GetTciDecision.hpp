// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"

namespace evolution::dg::subcell {
/// Returns the value of evolution::dg::subcell::Tags::TciDecision.
template <typename DbTagsList>
int get_tci_decision(const db::DataBox<DbTagsList>& box) {
  return db::get<Tags::TciDecision>(box);
}
}  // namespace evolution::dg::subcell
