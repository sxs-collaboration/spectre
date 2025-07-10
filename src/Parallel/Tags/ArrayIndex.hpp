// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace Parallel::Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag to retrieve the `ArrayIndex` from the DataBox.
template <typename Index>
struct ArrayIndex : db::SimpleTag {
  using type = Index;
};
}  // namespace Parallel::Tags
