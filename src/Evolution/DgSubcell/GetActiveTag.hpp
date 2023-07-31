// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"

namespace evolution::dg::subcell {
/// \brief Retrieve a tag from the active grid.
template <typename DgTag, typename SubcellTag, typename DbTagsList>
const typename DgTag::type& get_active_tag(const db::DataBox<DbTagsList>& box) {
  static_assert(
      std::is_same_v<typename DgTag::type, typename SubcellTag::type>);
  if (db::get<Tags::ActiveGrid>(box) == ActiveGrid::Subcell) {
    return db::get<SubcellTag>(box);
  }
  return db::get<DgTag>(box);
}

/// \brief Retrieve a tag from the inactive grid.
template <typename DgTag, typename SubcellTag, typename DbTagsList>
const typename DgTag::type& get_inactive_tag(
    const db::DataBox<DbTagsList>& box) {
  static_assert(
      std::is_same_v<typename DgTag::type, typename SubcellTag::type>);
  if (db::get<Tags::ActiveGrid>(box) == ActiveGrid::Subcell) {
    return db::get<DgTag>(box);
  }
  return db::get<SubcellTag>(box);
}
}  // namespace evolution::dg::subcell
