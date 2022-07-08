// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <unordered_set>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Parallel/Serialize.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criteria.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Criteria {
/// Option tags for AMR criteria
namespace OptionTags {
/// \ingroup OptionTagsGroup
/// Options for AMR criteria
struct Criteria {
  static constexpr Options::String help = "Options for AMR criteria";
  using type = std::vector<std::unique_ptr<amr::Criterion>>;
};
}  // namespace OptionTags

namespace Tags {
/// The set of adaptive mesh refinement criteria
struct Criteria : db::SimpleTag {
  using type = std::vector<std::unique_ptr<amr::Criterion>>;
  using option_tags = tmpl::list<amr::Criteria::OptionTags::Criteria>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) {
    return {deserialize<type>(serialize<type>(value).data())};
  }
};
}  // namespace Tags
}  // namespace amr::Criteria
