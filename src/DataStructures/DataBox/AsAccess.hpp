// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Access.hpp"
#include "DataStructures/DataBox/DataBox.hpp"

namespace db {
/// @{
/// \brief Convert a `db::DataBox` to a `db::Access`.
template <typename TagsList>
const Access& as_access(const DataBox<TagsList>& box) {
  return static_cast<const Access&>(box);
}

template <typename TagsList>
Access& as_access(DataBox<TagsList>& box) {
  return static_cast<Access&>(box);
}
/// @}
}  // namespace db
