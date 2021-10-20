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

/// Given a measurement, obtain a list of its submeasurements (i.e.,
/// `Measurement::submeasurements`).
/// @{
template <typename Measurement>
struct submeasurements {
  using type = typename Measurement::submeasurements;
};

template <typename Measurement>
using submeasurements_t = typename submeasurements<Measurement>::type;
/// @}

namespace detail {
template <typename Submeasurement, typename ControlSystems>
struct interpolation_target_tags_for_submeasurement {
 private:
  using declared_type =
      typename Submeasurement::template interpolation_target_tag<
          ControlSystems>;

 public:
  using type = tmpl::conditional_t<std::is_same_v<declared_type, void>,
                                   tmpl::list<>, tmpl::list<declared_type>>;
};
}  // namespace detail

/// Extract the `interpolation_target_tag` aliases from all
/// submeasurements for the list of control systems.  This is intended
/// for use in constructing the global list of interpolation target
/// tags in the metavariables.
template <typename ControlSystems>
using interpolation_target_tags = tmpl::flatten<tmpl::transform<
    measurements_t<ControlSystems>,
    tmpl::lazy::transform<
        submeasurements<tmpl::_1>,
        tmpl::defer<detail::interpolation_target_tags_for_submeasurement<
            tmpl::_1,
            control_systems_with_measurement<tmpl::pin<ControlSystems>,
                                             tmpl::parent<tmpl::_1>>>>>>>;
}  // namespace metafunctions
}  // namespace control_system
