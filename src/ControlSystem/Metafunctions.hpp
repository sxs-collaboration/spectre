// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <class Metavariables, typename ControlSystem>
struct ControlComponent;
/// \endcond

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

namespace detail {
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(component_being_mocked)

template <typename Metavariables>
using all_components = typename Metavariables::component_list;

template <typename Metavariables>
using all_not_mocked_components =
    tmpl::transform<all_components<Metavariables>,
                    get_component_being_mocked_or_default<tmpl::_1, tmpl::_1>>;
}  // namespace detail

/// Get all ControlComponent%s from the metavariables, even if they are mocked.
template <typename Metavariables>
using all_control_components =
    tmpl::filter<detail::all_not_mocked_components<Metavariables>,
                 tt::is_a<ControlComponent, tmpl::_1>>;

template <typename ControlSystems, typename Submeasurement>
struct event_from_submeasurement {
  using type = typename Submeasurement::template event<ControlSystems>;
};

template <typename ControlSystems, typename Submeasurement>
using event_from_submeasurement_t =
    typename event_from_submeasurement<ControlSystems, Submeasurement>::type;

namespace detail {
template <typename AllControlSystems, typename Measurement>
struct events_from_measurement {
  using submeasurements = submeasurements_t<Measurement>;
  using control_systems_with_measurement =
      control_systems_with_measurement_t<AllControlSystems, Measurement>;

  using type = tmpl::transform<
      submeasurements,
      event_from_submeasurement<tmpl::pin<control_systems_with_measurement>,
                                tmpl::_1>>;
};
template <typename AllControlSystems, typename Measurement>
using events_from_measurement_t =
    typename events_from_measurement<AllControlSystems, Measurement>::type;
}  // namespace detail

/// \ingroup ControlSystemGroup
/// The list of events needed for measurements for a list of control
/// systems.
template <typename ControlSystems>
using control_system_events = tmpl::flatten<tmpl::transform<
    metafunctions::measurements_t<ControlSystems>,
    detail::events_from_measurement<tmpl::pin<ControlSystems>, tmpl::_1>>>;
}  // namespace metafunctions
}  // namespace control_system
