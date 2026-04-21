// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <initializer_list>
#include <memory>
#include <type_traits>
#include <typeinfo>

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/LtsMode.hpp"
#include "Time/OptionTags/TimeStepper.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// The evolution TimeStepper.
///
/// The \p StepperType template parameter should be one of the time
/// stepper base classes, such as `TimeStepper` or `LtsTimeStepper`.
///
/// If the \p MonotonicLts parameter is true, the chosen stepper will
/// be required to be monotonic when parsed from options in local
/// time-stepping mode.  This is generally required for evolutions
/// with control systems.
///
/// For the contained object to be used, the reference tags listed in
/// `time_stepper_ref_tags<StepperType>` will also need to be added to
/// the DataBox.
template <typename StepperType, bool MonotonicLts = false>
struct ConcreteTimeStepper : db::SimpleTag {
  using type = std::unique_ptr<StepperType>;
  template <typename Metavars>
  using option_tags = tmpl::list<::OptionTags::TimeStepper<StepperType>>;

  static constexpr bool pass_metavariables = true;
  template <typename Metavars>
  static std::unique_ptr<StepperType> create_from_options(
      const std::unique_ptr<StepperType>& time_stepper) {
    const ::LtsMode lts_mode = Metavars::local_time_stepping
                                   ? ::LtsMode::Conservative
                                   : ::LtsMode::Off;
    using factory_types =
        tmpl::at<typename Metavars::factory_creation::factory_classes,
                 StepperType>;
    [&]<typename... Derived>(tmpl::list<Derived...> /*meta*/) {
      validate_time_stepper(*time_stepper, lts_mode, {&typeid(Derived)...});
    }(factory_types{});
    return serialize_and_deserialize<type>(time_stepper);
  }

 private:
  static void validate_time_stepper(
      const StepperType& time_stepper, ::LtsMode lts_mode,
      std::initializer_list<const std::type_info*> factory_types);
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
template <typename StepperInterface, typename StepperType,
          typename MonotonicLts>
struct TimeStepperRef : TimeStepper<StepperInterface>, db::ReferenceTag {
  using base = TimeStepper<StepperInterface>;
  using argument_tags =
      tmpl::list<ConcreteTimeStepper<StepperType, MonotonicLts::value>>;
  static const StepperInterface& get(const StepperType& stepper) {
    return stepper;
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// Compute tag to allow LTS code to compile in GTS executables.
struct LtsOrError : TimeStepper<LtsTimeStepper>, db::ReferenceTag {
  using base = TimeStepper<LtsTimeStepper>;
  using argument_tags = tmpl::list<TimeStepper<::TimeStepper>>;
  static const LtsTimeStepper& get(const ::TimeStepper& stepper);
};
}  // namespace Tags

/// \ingroup TimeGroup
/// List of immutable tags needed when adding a Tags::ConcreteTimeStepper.
template <typename StepperType, bool MonotonicLts = false>
using time_stepper_ref_tags = tmpl::append<
    tmpl::transform<
        typename StepperType::provided_time_stepper_interfaces,
        tmpl::bind<::Tags::TimeStepperRef, tmpl::_1, tmpl::pin<StepperType>,
                   std::bool_constant<MonotonicLts>>>,
    tmpl::conditional_t<
        tmpl::list_contains_v<
            typename StepperType::provided_time_stepper_interfaces,
            LtsTimeStepper>,
        tmpl::list<>, tmpl::list<Tags::LtsOrError>>>;
