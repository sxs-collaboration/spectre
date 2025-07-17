// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"

namespace Parallel::Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag to retrieve the `Metavariables` from the DataBox.
struct Metavariables : db::BaseTag {};

template <typename Metavars>
struct MetavariablesImpl : Metavariables, db::SimpleTag {
  using type = Metavars;
  static std::string name() { return "Metavariables"; }
};
}  // namespace Parallel::Tags
