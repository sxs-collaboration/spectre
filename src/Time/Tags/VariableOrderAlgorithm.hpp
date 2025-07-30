// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/OptionTags/VariableOrderAlgorithm.hpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Algorithm for changing the time-stepper order in a
/// variable-order evolution.
/// \see ChangeTimeStepperOrder
struct VariableOrderAlgorithm : db::SimpleTag {
  using type = ::VariableOrderAlgorithm;
  using option_tags = tmpl::list<::OptionTags::VariableOrderAlgorithm>;
  static constexpr bool is_overlayable = true;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};
}  // namespace Tags
