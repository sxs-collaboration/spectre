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
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"
#include "Options/Options.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/History.hpp"
#include "Time/StepChoosers/StepChooser.hpp"        // IWYU pragma: keep
#include "Time/StepControllers/StepController.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for ::TimeId for the algorithm state
struct TimeId : db::SimpleTag {
  static std::string name() noexcept { return "TimeId"; }
  using type = ::TimeId;
  template <typename Tag>
  using step_prefix = typename Tags::dt<Tag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for step size
struct TimeStep : db::SimpleTag {
  static std::string name() noexcept { return "TimeStep"; }
  using type = ::TimeDelta;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for compute item for current ::Time (from TimeId)
struct Time : db::ComputeTag {
  static std::string name() noexcept { return "Time"; }
  static auto function(const ::TimeId& id) noexcept { return id.time(); }
  using argument_tags = tmpl::list<TimeId>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Prefix for TimeStepper history
///
/// \tparam Tag tag for the variables
/// \tparam DtTag tag for the time derivative of the variables
template <typename Tag, typename DtTag>
struct HistoryEvolvedVariables : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept { return "HistoryEvolvedVariables"; }
  using tag = Tag;
  using type = TimeSteppers::History<db::item_type<Tag>, db::item_type<DtTag>>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// Tag for TimeStepper boundary history
template <typename LocalVars, typename RemoteVars, typename CouplingResult>
struct BoundaryHistory : db::SimpleTag {
  static std::string name() noexcept { return "BoundaryHistory"; }
  using type =
      TimeSteppers::BoundaryHistory<LocalVars, RemoteVars, CouplingResult>;
};

}  // namespace Tags

namespace OptionTags {

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The final time
struct FinalTime {
  using type = double;
  static constexpr OptionString help{"The final time"};
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The ::TimeStepper
struct TimeStepper {};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The ::TimeStepper, specifying a (base) type.  Can be
/// retrieved through OptionTags::TimeStepper.
template <typename StepperType>
struct TypedTimeStepper : TimeStepper {
  static std::string name() noexcept { return "TimeStepper"; }
  static constexpr OptionString help{"The time stepper"};
  using type = std::unique_ptr<StepperType>;
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
template <typename Registrars>
struct StepChoosers {
  static constexpr OptionString help{"Limits on LTS step size"};
  using type = std::vector<std::unique_ptr<::StepChooser<Registrars>>>;
  static size_t lower_bound_on_size() noexcept { return 1; }
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
struct StepController {
  static constexpr OptionString help{"The LTS step controller"};
  using type = std::unique_ptr<::StepController>;
};

}  // namespace OptionTags

namespace OptionTags {

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The time at which to start the simulation
struct InitialTime {
  using type = double;
  static constexpr OptionString help = {
      "The time at which the evolution is started."};
  static type default_value() noexcept { return 0.0; }
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The initial time step taken by the time stepper. This may be
/// overridden by an adaptive stepper
struct InitialTimeStep {
  using type = double;
  static constexpr OptionString help =
      "The initial time step, before local stepping adjustment";
};

/// \ingroup OptionTagsGroup
/// \ingroup TimeGroup
/// \brief The initial slab size
struct InitialSlabSize {
  using type = double;
  static constexpr OptionString help = "The initial slab size";
  static type lower_bound() noexcept { return 0.; }
};


/// \ingroup OptionTagsGroup
/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Contains the events and triggers
template <typename EventRegistrars, typename TriggerRegistrars>
struct EventsAndTriggers : EventsAndTriggersTagBase {
  using type = ::EventsAndTriggers<EventRegistrars, TriggerRegistrars>;
  static constexpr OptionString help =
      "Events and triggers to run at slab boundaries";
};
}  // namespace OptionTags
