// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ParallelAlgorithms/Events/ObserveTimeStepVolume.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/FloatingPointType.hpp"
#include "IO/H5/TensorData.hpp"
#include "ParallelAlgorithms/Events/ObserveConstantsPerElement.hpp"
#include "Time/History.hpp"
#include "Time/Time.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t VolumeDim>
class Domain;
template <size_t VolumeDim>
class ElementId;
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
/// \endcond

namespace dg::Events {
template <typename System>
ObserveTimeStepVolume<System>::ObserveTimeStepVolume(CkMigrateMessage* const m)
    : ObserveConstantsPerElement<volume_dim>(m) {}

template <typename System>
ObserveTimeStepVolume<System>::ObserveTimeStepVolume() = default;

template <typename System>
ObserveTimeStepVolume<System>::ObserveTimeStepVolume(
    const std::string& subfile_name,
    const ::FloatingPointType coordinates_floating_point_type,
    const ::FloatingPointType floating_point_type)
    : ObserveConstantsPerElement<volume_dim>(
          subfile_name, coordinates_floating_point_type, floating_point_type) {}

template <typename System>
bool ObserveTimeStepVolume<System>::needs_evolved_variables() const {
  return false;
}

template <typename System>
std::vector<TensorComponent> ObserveTimeStepVolume<System>::assemble_data(
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const Domain<volume_dim>& domain, const ElementId<volume_dim>& element_id,
    const TimeDelta& time_step, const double minimum_grid_spacing,
    const TimeSteppers::History<typename System::variables_tag::type>& history)
    const {
  std::vector<TensorComponent> components = this->allocate_and_insert_coords(
      4, time, functions_of_time, domain, element_id);
  this->add_constant(make_not_null(&components), "Time step",
                     time_step.value());
  this->add_constant(make_not_null(&components), "Slab fraction",
                     time_step.fraction().value());
  this->add_constant(make_not_null(&components), "Minimum grid spacing",
                     minimum_grid_spacing);
  this->add_constant(make_not_null(&components), "Integration order",
                     static_cast<double>(history.integration_order()));
  return components;
}

/// \cond
template <typename System>
PUP::able::PUP_ID ObserveTimeStepVolume<System>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace dg::Events
