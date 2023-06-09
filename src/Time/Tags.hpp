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
#include "DataStructures/LinkedMessageId.hpp"
#include "Evolution/Tags.hpp"
#include "Options/String.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/History.hpp"
#include "Time/StepChoosers/StepChooser.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace evolution::Tags {
struct PreviousTriggerTime;
}  // namespace evolution::Tags
/// \endcond

namespace OptionTags {

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
template <typename StepperType>
struct TimeStepper {
  static std::string name() { return "TimeStepper"; }
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
  static size_t lower_bound_on_size() { return 1; }
  using group = evolution::OptionTags::Group;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The time at which to start the simulation
struct InitialTime {
  using type = double;
  static constexpr Options::String help = {
      "The time at which the evolution is started."};
  static type suggested_value() { return 0.0; }
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
  static type lower_bound() { return 0.; }
  using group = evolution::OptionTags::Group;
};
}  // namespace OptionTags

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
/// \brief Tag for the current time as a double
///
/// The meaning of "current time" varies during the algorithm, but
/// generally is whatever time is appropriate for the calculation
/// being run.  Usually this is the substep time, but things such as
/// dense-output calculations may temporarily change the value.
struct Time : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::InitialTime>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double initial_time) {
    return initial_time;
  }
};

/// @{
/// \ingroup TimeGroup
/// \brief Tag for the current and previous time as doubles
///
/// \warning The previous time is calculated via the value of the
/// ::evolution::Tags::PreviousTriggerTime. Therefore, this tag can only be
/// used in the context of dense triggers as that is where the
/// ::evolution::Tags::PreviousTriggerTime tag is set. Any Events that request
/// this tag in their `argument_tags` type alias, must be triggered by a
/// DenseTrigger.
///
/// \note The Index is just so we can have multiple of this tag in the same
/// DataBox.
template <size_t Index>
struct TimeAndPrevious : db::SimpleTag {
  using type = LinkedMessageId<double>;
  static std::string name() { return "TimeAndPrevious" + get_output(Index); }
};

template <size_t Index>
struct TimeAndPreviousCompute : TimeAndPrevious<Index>, db::ComputeTag {
  using argument_tags =
      tmpl::list<::Tags::Time, ::evolution::Tags::PreviousTriggerTime>;
  using base = TimeAndPrevious<Index>;
  using return_type = LinkedMessageId<double>;

  static void function(
      gsl::not_null<LinkedMessageId<double>*> time_and_previous,
      const double time, const std::optional<double>& previous_time) {
    time_and_previous->id = time;
    time_and_previous->previous = previous_time;
  }
};
/// @}

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

template <typename Tag>
struct HistoryEvolvedVariables : HistoryEvolvedVariables<>, db::SimpleTag {
  using type = TimeSteppers::History<typename Tag::type>;
};
/// \endcond

/// \ingroup TimeGroup
/// From a list of tags `TagList`, extract all tags that are template
/// specializations of `HistoryEvolvedVariables`.
template <typename TagList>
using get_all_history_tags =
    tmpl::filter<TagList, tt::is_a<::Tags::HistoryEvolvedVariables, tmpl::_1>>;

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for the stepper error measure.
template <typename Tag>
struct StepperError : db::PrefixTag, db::SimpleTag {
  static std::string name() {
    return "StepperError(" + db::tag_name<Tag>() + ")";
  }
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for the previous value of the stepper error measure.
template <typename Tag>
struct PreviousStepperError : db::PrefixTag, db::SimpleTag {
  static std::string name() {
    return "PreviousStepperError(" + db::tag_name<Tag>() + ")";
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

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for a vector of ::StepChooser%s
struct StepChoosers : db::SimpleTag {
  using type =
      std::vector<std::unique_ptr<::StepChooser<StepChooserUse::LtsStep>>>;
  using option_tags = tmpl::list<::OptionTags::StepChoosers>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& step_choosers) {
    return deserialize<type>(serialize<type>(step_choosers).data());
  }
};

/// \ingroup TimeGroup
/// \brief Tag for reporting whether the `ErrorControl` step chooser is in
/// use.
struct IsUsingTimeSteppingErrorControl : db::SimpleTag {
  using type = bool;
};
}  // namespace Tags
