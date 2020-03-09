// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to Time quantities

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
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
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

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
  static auto function(const ::TimeStepId& id) noexcept {
    return id.substep_time();
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

template <typename Tag>
struct HistoryEvolvedVariables : HistoryEvolvedVariables<>, db::SimpleTag {
  using type = TimeSteppers::History<
      db::const_item_type<Tag>,
      db::const_item_type<db::add_tag_prefix<Tags::dt, Tag>>>;
};
/// \endcond

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
  static constexpr OptionString help{"The time stepper"};
  using type = std::unique_ptr<StepperType>;
  using group = evolution::OptionTags::Group;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
template <typename Registrars>
struct StepChoosers {
  static constexpr OptionString help{"Limits on LTS step size"};
  using type = std::vector<std::unique_ptr<::StepChooser<Registrars>>>;
  static size_t lower_bound_on_size() noexcept { return 1; }
  using group = evolution::OptionTags::Group;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
struct StepController {
  static constexpr OptionString help{"The LTS step controller"};
  using type = std::unique_ptr<::StepController>;
  using group = evolution::OptionTags::Group;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The time at which to start the simulation
struct InitialTime {
  using type = double;
  static constexpr OptionString help = {
      "The time at which the evolution is started."};
  static type default_value() noexcept { return 0.0; }
  using group = evolution::OptionTags::Group;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The initial time step taken by the time stepper. This may be
/// overridden by an adaptive stepper
struct InitialTimeStep {
  using type = double;
  static constexpr OptionString help =
      "The initial time step, before local stepping adjustment";
  using group = evolution::OptionTags::Group;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The initial slab size
struct InitialSlabSize {
  using type = double;
  static constexpr OptionString help = "The initial slab size";
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
template <typename Registrars>
struct StepChoosers : db::SimpleTag {
  using type = std::vector<std::unique_ptr<::StepChooser<Registrars>>>;
  using option_tags = tmpl::list<::OptionTags::StepChoosers<Registrars>>;

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
}  // namespace Tags
