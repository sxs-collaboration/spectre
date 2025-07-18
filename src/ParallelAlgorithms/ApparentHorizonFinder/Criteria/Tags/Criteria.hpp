// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Criterion.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::Criteria {
/// Option tags for adaptive horizon finding criteria
namespace OptionTags {
/// \ingroup OptionTagsGroup
/// Options for adaptive horizon finding criteria
struct Criteria {
  static constexpr Options::String help =
      "Options for adaptive horizon finding criteria";
  using type = std::vector<std::unique_ptr<ah::Criterion>>;
};
}  // namespace OptionTags

namespace Tags {
/// The set of adaptive horizon finding criteria
struct Criteria : db::SimpleTag {
  using type = std::vector<std::unique_ptr<ah::Criterion>>;
  using option_tags = tmpl::list<ah::Criteria::OptionTags::Criteria>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) {
    return {deserialize<type>(serialize<type>(value).data())};
  }
};
}  // namespace Tags
}  // namespace ah::Criteria
