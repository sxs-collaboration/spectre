// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Tags.hpp"
#include "Options/String.hpp"
#include "Time/VariableOrderAlgorithm.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief Algorithm for changing the time-stepper order in a
/// variable-order evolution.
/// \see ChangeTimeStepperOrder
struct VariableOrderAlgorithm {
  using type = ::VariableOrderAlgorithm;
  static constexpr Options::String help =
      "Algorithm for changing the time-stepper order in a variable-order "
      "evolution.";
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags
