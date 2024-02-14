// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace amr {
/// Option tags for AMR polocies
namespace OptionTags {
/// \ingroup OptionTagsGroup
/// Options for AMR policies
struct Policies {
  static constexpr Options::String help = "Options for AMR policies";
  using type = amr::Policies;
  using group = amr::OptionTags::AmrGroup;
};
}  // namespace OptionTags

namespace Tags {
/// The policies for adaptive mesh refinement
struct Policies : db::SimpleTag {
  using type = amr::Policies;
  using option_tags = tmpl::list<amr::OptionTags::Policies>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) { return value; }
};
}  // namespace Tags
}  // namespace amr
