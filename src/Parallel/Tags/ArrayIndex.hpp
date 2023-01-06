// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace Parallel::Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag to retrieve the `ArrayIndex` from the DataBox.
struct ArrayIndex : db::BaseTag {};

template <typename Index>
struct ArrayIndexImpl : ArrayIndex, db::SimpleTag {
  using base = ArrayIndex;
  using type = Index;
};
}  // namespace Parallel::Tags
