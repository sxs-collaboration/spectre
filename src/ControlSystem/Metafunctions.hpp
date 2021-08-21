// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"

namespace control_system {
/// \ingroup ControlSystemGroup
/// Metafunctions associated with the control systems
namespace metafunctions {
/// Extract the `measurement` alias from a control system struct.
template <typename ControlSystem>
struct measurement {
  using type = typename ControlSystem::measurement;
};

/// Given a list of control systems, obtain a list of distinct control
/// system measurement structs used by them.
/// @{
template <typename ControlSystems>
struct measurements {
  using type = tmpl::remove_duplicates<
      tmpl::transform<ControlSystems, measurement<tmpl::_1>>>;
};

template <typename ControlSystems>
using measurements_t = typename measurements<ControlSystems>::type;
/// @}

/// Given a list of control systems, extract those using a given
/// measurement.
/// @{
template <typename ControlSystems, typename Measurement>
struct control_systems_with_measurement
    : tmpl::lazy::filter<ControlSystems, std::is_same<measurement<tmpl::_1>,
                                                      tmpl::pin<Measurement>>> {
};

template <typename ControlSystems, typename Measurement>
using control_systems_with_measurement_t =
    typename control_systems_with_measurement<ControlSystems,
                                              Measurement>::type;
/// @}
}  // namespace metafunctions
}  // namespace control_system
