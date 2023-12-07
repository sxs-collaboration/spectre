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
/// The evolution TimeStepper.  The template parameter should be one
/// of the time stepper base classes, such as `TimeStepper` or
/// `LtsTimeStepper`.
///
/// For the contained object to be used, the reference tags listed in
/// `time_stepper_ref_tags<StepperType>` will also need to be added to
/// the DataBox.
template <typename StepperType>
struct ConcreteTimeStepper : db::SimpleTag {
  using type = std::unique_ptr<StepperType>;
  using option_tags = tmpl::list<::OptionTags::TimeStepper<StepperType>>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<StepperType> create_from_options(
      const std::unique_ptr<StepperType>& time_stepper) {
    return deserialize<type>(serialize<type>(time_stepper).data());
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// Access to a time stepper through the `StepperInterface` interface
/// (such as `TimeStepper` or `LtsTimeStepper`).
///
/// \details This tag cannot be added directly to the DataBox of
/// GlobalCache because it contains an abstract type, but can only be
/// used for retrieving the time stepper.  Instead, the
/// `ConcreteTimeStepper` tag should be added, along with the
/// reference tags given by `time_stepper_ref_tags`.
template <typename StepperInterface>
struct TimeStepper : db::SimpleTag {
  using type = StepperInterface;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// Reference tag to provide access to the time stepper through its
/// provided interfaces, such as `Tags::TimeStepper<TimeStepper>` and
/// `Tags::TimeStepper<LtsTimeStepper>`.  Usually added through the
/// `time_stepper_ref_tags` alias.
template <typename StepperInterface, typename StepperType>
struct TimeStepperRef : TimeStepper<StepperInterface>, db::ReferenceTag {
  using base = TimeStepper<StepperInterface>;
  using argument_tags = tmpl::list<ConcreteTimeStepper<StepperType>>;
  static const StepperInterface& get(const StepperType& stepper) {
    return stepper;
  }
};
}  // namespace Tags

/// \ingroup TimeGroup
/// List of Tags::TimeStepperRef specializations needed when adding a
/// Tags::ConcreteTimeStepper.
template <typename StepperType>
using time_stepper_ref_tags = tmpl::transform<
    typename StepperType::provided_time_stepper_interfaces,
    tmpl::bind<::Tags::TimeStepperRef, tmpl::_1, tmpl::pin<StepperType>>>;
