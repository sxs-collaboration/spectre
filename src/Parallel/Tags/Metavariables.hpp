// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace Parallel::Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag to retrieve the `Metavariables` from the DataBox.
struct Metavariables : db::BaseTag {};

template <typename Metavars>
struct MetavariablesImpl : Metavariables, db::SimpleTag {
  using base = Metavariables;
  using type = Metavars;
};
}  // namespace Parallel::Tags
