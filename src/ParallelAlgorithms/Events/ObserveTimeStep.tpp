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
template <typename System>
ObserveTimeStep<System>::ObserveTimeStep(CkMigrateMessage* const m)
    : Event(m) {}

template <typename System>
ObserveTimeStep<System>::ObserveTimeStep() = default;

template <typename System>
ObserveTimeStep<System>::ObserveTimeStep(const std::string& subfile_name,
                                         const bool output_time,
                                         const bool observe_per_core)
    : subfile_path_("/" + subfile_name),
      output_time_(output_time),
      observe_per_core_(observe_per_core) {}

template <typename System>
std::pair<observers::TypeOfObservation, observers::ObservationKey>
ObserveTimeStep<System>::get_observation_type_and_key_for_registration() const {
  return {observers::TypeOfObservation::Reduction,
          observers::ObservationKey{subfile_path_ + ".dat"}};
}

template <typename System>
bool ObserveTimeStep<System>::needs_evolved_variables() const {
  return false;
}

template <typename System>
void ObserveTimeStep<System>::pup(PUP::er& p) {
  Event::pup(p);
  p | subfile_path_;
  p | output_time_;
  p | observe_per_core_;
}

template <typename System>
auto ObserveTimeStep<System>::assemble_data(
    const TimeDelta& time_step,
    const typename System::variables_tag::type& variables,
    const ObservationValue& observation_value) const
    -> std::tuple<observers::ObservationId, std::vector<std::string>,
                  ReductionData,
                  std::optional<Events::detail::FormatTimeOutput>> {
  // For empty vars, we use 1 grid point to avoid divisions by zero
  // after reduction.
  size_t number_of_grid_points = 1;
  if constexpr (not std::is_same_v<typename System::variables_tag::tags_list,
                                   tmpl::list<>>) {
    number_of_grid_points = variables.number_of_grid_points();
  }
  const double slab_size = time_step.slab().duration().value();
  const double step_size = abs(time_step.value());
  const double wall_time = sys::wall_time();

  auto formatter = output_time_
                       ? std::make_optional(Events::detail::FormatTimeOutput{})
                       : std::nullopt;

  observers::ObservationId observation_id{observation_value.value,
                                          subfile_path_ + ".dat"};
  std::vector<std::string> legend{
      observation_value.name, "NumberOfPoints",    "Slab size",
      "Minimum time step",    "Maximum time step", "Effective time step",
      "Minimum Walltime",     "Maximum Walltime"};
  ReductionData reduction_data{observation_value.value,
                               number_of_grid_points,
                               slab_size,
                               step_size,
                               step_size,
                               number_of_grid_points / step_size,
                               wall_time,
                               wall_time};

  return {std::move(observation_id), std::move(legend),
          std::move(reduction_data), std::move(formatter)};
}

/// \cond
#ifndef __CUDA_ARCH__
template <typename System>
PUP::able::PUP_ID ObserveTimeStep<System>::my_PUP_ID = 0;  // NOLINT
#endif                                                     // __CUDA_ARCH__
/// \endcond
}  // namespace Events
