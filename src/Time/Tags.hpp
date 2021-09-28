// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to Time quantities

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Serialize.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/History.hpp"
#include "Time/StepChoosers/StepChooser.hpp"        // IWYU pragma: keep
#include "Time/StepControllers/StepController.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Time/TimeAndPrevious.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace evolution {
namespace Tags {
struct PreviousTriggerTime;
}  // namespace Tags
}  // namespace evolution
/// \endcond

namespace Tags {

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for ::TimeStepId for the algorithm state
struct TimeStepId : db::SimpleTag {
  using type = ::TimeStepId;
  template <typename Tag>
  using step_prefix = typename Tags::dt<Tag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for step size
struct TimeStep : db::SimpleTag {
  using type = ::TimeDelta;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for ::Time of the current substep
///
/// \see SubstepTimeCompute
struct SubstepTime : db::SimpleTag {
  using type = ::Time;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for computing the substep time from (from `Tags::TimeStepId`)
///
/// \see SubstepTime
struct SubstepTimeCompute : SubstepTime, db::ComputeTag {
  using base = SubstepTime;
  using return_type = typename base::type;
  static void function(const gsl::not_null<return_type*> substep_time,
                       const ::TimeStepId& id) noexcept {
    *substep_time = id.substep_time();
  }
  using argument_tags = tmpl::list<TimeStepId>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for the current time as a double
struct Time : db::SimpleTag {
  using type = double;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// Tag for the TimeStepper history
///
/// Leaving the template parameter unspecified gives a base tag.
///
/// \tparam Tag tag for the variables
template <typename Tag = void>
struct HistoryEvolvedVariables;

/// \cond
template <>
struct HistoryEvolvedVariables<> : db::BaseTag {};

template <typename TagsList>
struct HistoryEvolvedVariables<::Tags::Variables<TagsList>>
    : HistoryEvolvedVariables<>, db::SimpleTag {
  using type =
      TimeSteppers::History<::Variables<TagsList>,
                            ::Variables<db::wrap_tags_in<Tags::dt, TagsList>>>;
};

template <typename Tag>
struct HistoryEvolvedVariables : HistoryEvolvedVariables<>, db::SimpleTag {
  using type =
      TimeSteppers::History<typename Tag::type, typename Tags::dt<Tag>::type>;
};
/// \endcond

/// \ingroup TimeGroup
/// From a list of tags `TagList`, extract all tags that are template
/// specializations of `HistoryEvolvedVariables`.
template <typename TagList>
using get_all_history_tags =
    tmpl::filter<TagList,
                 tt::is_a_lambda<::Tags::HistoryEvolvedVariables, tmpl::_1>>;

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for the stepper error measure.
template <typename Tag>
struct StepperError : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    return "StepperError(" + db::tag_name<Tag>() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup TimeGroup
/// \brief Tag indicating whether the stepper error has been updated on the
/// current step
struct StepperErrorUpdated : db::SimpleTag {
  using type = bool;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// Tag for TimeStepper boundary history
template <typename LocalVars, typename RemoteVars, typename CouplingResult>
struct BoundaryHistory : db::SimpleTag {
  using type =
      TimeSteppers::BoundaryHistory<LocalVars, RemoteVars, CouplingResult>;
};

}  // namespace Tags

namespace OptionTags {

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
template <typename StepperType>
struct TimeStepper {
  static std::string name() noexcept { return "TimeStepper"; }
  static constexpr Options::String help{"The time stepper"};
  using type = std::unique_ptr<StepperType>;
  using group = evolution::OptionTags::Group;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
struct StepChoosers {
  static constexpr Options::String help{"Limits on LTS step size"};
  using type =
      std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>;
  static size_t lower_bound_on_size() noexcept { return 1; }
  using group = evolution::OptionTags::Group;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
struct StepController {
  static constexpr Options::String help{"The LTS step controller"};
  using type = std::unique_ptr<::StepController>;
  using group = evolution::OptionTags::Group;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The time at which to start the simulation
struct InitialTime {
  using type = double;
  static constexpr Options::String help = {
      "The time at which the evolution is started."};
  static type suggested_value() noexcept { return 0.0; }
  using group = evolution::OptionTags::Group;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The initial time step taken by the time stepper. This may be
/// overridden by an adaptive stepper
struct InitialTimeStep {
  using type = double;
  static constexpr Options::String help =
      "The initial time step, before local stepping adjustment";
  using group = evolution::OptionTags::Group;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The initial slab size
struct InitialSlabSize {
  using type = double;
  static constexpr Options::String help = "The initial slab size";
  static type lower_bound() noexcept { return 0.; }
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags

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
      const std::unique_ptr<StepperType>& time_stepper) noexcept {
    return deserialize<type>(serialize<type>(time_stepper).data());
  }
};
/// \endcond

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for a vector of ::StepChooser%s
struct StepChoosers : db::SimpleTag {
  using type =
      std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>;
  using option_tags = tmpl::list<::OptionTags::StepChoosers>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& step_choosers) noexcept {
    return deserialize<type>(serialize<type>(step_choosers).data());
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for a ::StepController
struct StepController : db::SimpleTag {
  using type = std::unique_ptr<::StepController>;
  using option_tags = tmpl::list<::OptionTags::StepController>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<::StepController> create_from_options(
      const std::unique_ptr<::StepController>& step_controller) noexcept {
    return deserialize<type>(serialize<type>(step_controller).data());
  }
};

struct TimeAndPrevious : db::SimpleTag {
  using type = ::TimeAndPrevious;
};

struct TimeAndPreviousCompute : db::ComputeTag, TimeAndPrevious {
  using base = TimeAndPrevious;
  using return_type = ::TimeAndPrevious;
  static void function(
      const gsl::not_null<::TimeAndPrevious*> time_and_previous,
      const double time, const std::optional<double>& previous) noexcept {
    time_and_previous->time = time;
    time_and_previous->previous_time = previous;
  }
  using argument_tags =
      tmpl::list<::Tags::Time, ::evolution::Tags::PreviousTriggerTime>;
};
}  // namespace Tags
