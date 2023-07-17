// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/OptionTags/TimeStepper.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for a ::TimeStepper of type `StepperType`.
///
/// Leaving the template parameter unspecified gives a base tag.
template <typename StepperType = void>
struct TimeStepper;

/// \cond
template <>
struct TimeStepper<> : db::BaseTag {};

template <typename StepperType>
struct TimeStepper : TimeStepper<>, db::SimpleTag {
  using type = std::unique_ptr<StepperType>;
  using option_tags = tmpl::list<::OptionTags::TimeStepper<StepperType>>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<StepperType> create_from_options(
      const std::unique_ptr<StepperType>& time_stepper) {
    return deserialize<type>(serialize<type>(time_stepper).data());
  }
};
/// \endcond
}  // namespace Tags
