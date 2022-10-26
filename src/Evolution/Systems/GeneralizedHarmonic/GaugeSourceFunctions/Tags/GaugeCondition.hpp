// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Options/Options.hpp"

/// \cond
namespace GeneralizedHarmonic::OptionTags {
struct Group;
}  // namespace GeneralizedHarmonic::OptionTags
/// \endcond

namespace GeneralizedHarmonic::gauges {
namespace OptionTags {
struct GaugeCondition {
  using type = std::unique_ptr<gauges::GaugeCondition>;
  static constexpr Options::String help{"The gauge condition to impose."};
  using group = GeneralizedHarmonic::OptionTags::Group;
};
}  // namespace OptionTags

namespace Tags {
/// \brief The gauge condition to impose.
struct GaugeCondition : db::SimpleTag {
  using type = std::unique_ptr<gauges::GaugeCondition>;
  using option_tags =
      tmpl::list<GeneralizedHarmonic::gauges::OptionTags::GaugeCondition>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<gauges::GaugeCondition> create_from_options(
      const std::unique_ptr<gauges::GaugeCondition>& gauge_condition) {
    return gauge_condition->get_clone();
  }
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic::gauges
