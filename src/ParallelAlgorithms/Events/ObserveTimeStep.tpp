// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ParallelAlgorithms/Events/ObserveTimeStep.hpp"

#include <cmath>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Time.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"

namespace Events {
template <typename System, typename... VariablesTags>
ObserveTimeStep<System, tmpl::list<VariablesTags...>>::ObserveTimeStep(
    CkMigrateMessage* const m)
    : Event(m) {}

template <typename System, typename... VariablesTags>
ObserveTimeStep<System, tmpl::list<VariablesTags...>>::ObserveTimeStep() =
    default;

template <typename System, typename... VariablesTags>
ObserveTimeStep<System, tmpl::list<VariablesTags...>>::ObserveTimeStep(
    const std::string& subfile_name, const bool output_time,
    const bool observe_per_core)
    : subfile_path_("/" + subfile_name),
      output_time_(output_time),
      observe_per_core_(observe_per_core) {}

template <typename System, typename... VariablesTags>
std::pair<observers::TypeOfObservation, observers::ObservationKey>
ObserveTimeStep<System, tmpl::list<VariablesTags...>>::
    get_observation_type_and_key_for_registration() const {
  return {observers::TypeOfObservation::Reduction,
          observers::ObservationKey{subfile_path_ + ".dat"}};
}

template <typename System, typename... VariablesTags>
bool ObserveTimeStep<
    System, tmpl::list<VariablesTags...>>::needs_evolved_variables() const {
  return false;
}

template <typename System, typename... VariablesTags>
void ObserveTimeStep<System, tmpl::list<VariablesTags...>>::pup(PUP::er& p) {
  Event::pup(p);
  p | subfile_path_;
  p | output_time_;
  p | observe_per_core_;
}

template <typename System, typename... VariablesTags>
auto ObserveTimeStep<System, tmpl::list<VariablesTags...>>::assemble_data(
    const TimeDelta& time_step,
    const typename VariablesTags::type&... variables,
    const ObservationValue& observation_value) const
    -> std::tuple<observers::ObservationId, std::vector<std::string>,
                  ReductionData,
                  std::optional<Events::detail::FormatTimeOutput>> {
  size_t number_of_degrees_of_freedom = (... + variables.size());
  // For empty vars, we use 1 grid point to avoid divisions by zero
  // after reduction.
  if (number_of_degrees_of_freedom == 0) {
    number_of_degrees_of_freedom = 1;
  }
  const double slab_size = time_step.slab().duration().value();
  const double step_size = abs(time_step.value());
  const double wall_time = sys::wall_time();

  auto formatter = output_time_
                       ? std::make_optional(Events::detail::FormatTimeOutput{})
                       : std::nullopt;

  observers::ObservationId observation_id{observation_value.value,
                                          subfile_path_ + ".dat"};
  std::vector<std::string> legend{observation_value.name,
                                  "Number of degrees of freedom",
                                  "Slab size",
                                  "Minimum time step",
                                  "Maximum time step",
                                  "Effective time step",
                                  "Minimum Walltime",
                                  "Maximum Walltime"};
  ReductionData reduction_data{observation_value.value,
                               number_of_degrees_of_freedom,
                               slab_size,
                               step_size,
                               step_size,
                               number_of_degrees_of_freedom / step_size,
                               wall_time,
                               wall_time};

  return {std::move(observation_id), std::move(legend),
          std::move(reduction_data), std::move(formatter)};
}

/// \cond
template <typename System, typename... VariablesTags>
PUP::able::PUP_ID
    ObserveTimeStep<System, tmpl::list<VariablesTags...>>::my_PUP_ID =
        0;  // NOLINT
/// \endcond
}  // namespace Events
