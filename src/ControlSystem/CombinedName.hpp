// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <vector>

#include "ControlSystem/Metafunctions.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace control_system {
/*!
 * \brief Given a `tmpl::list` of control systems, returns a string which is a
 * sorted concatenation of all the control systems' names.
 *
 * \details As an example, if `ListOfControlSystems` is `tmpl::list<Rotation,
 * Expansion, Translation>`, this function would return
 * `ExpansionRotationTranslation`. Sorting is done using `std::sort`.
 *
 * \tparam ListOfControlSystems `tmpl::list` of control systems
 */
template <typename ListOfControlSystems>
std::string combined_name() {
  std::vector<std::string> control_system_names{};
  control_system_names.reserve(tmpl::size<ListOfControlSystems>::value);
  std::string combined_name{};

  tmpl::for_each<ListOfControlSystems>(
      [&control_system_names](auto control_system_v) {
        using control_system = tmpl::type_from<decltype(control_system_v)>;
        control_system_names.emplace_back(control_system::name());
      });

  alg::sort(control_system_names);

  for (const std::string& name : control_system_names) {
    combined_name += name;
  }

  return combined_name;
}

/*!
 * \brief Given a `tmpl::list` of control systems, this returns a map between
 * the name of each control system, and the `control_system::combined_name` of
 * all control systems with the same measurement.
 *
 * \details All control systems have a type alias `measurement` corresponding to
 * a `control_system::protocols::Measurement`. Some control systems can have the
 * same measurement, while some have different ones. This function splits the
 * template list of control systems into a `tmpl::list` of `tmpl::list`s where
 * each inner list holds control systems with the same measurement. Each of
 * these inner lists is used in `control_system::combined_name` to get the
 * concatenated name for those control systems. This combined name is then used
 * as the value in the resulting map for they key of each control system in the
 * inner list.
 *
 * \tparam ControlSystems `tmpl::list` of control systems
 */
template <typename ControlSystems>
std::unordered_map<std::string, std::string> system_to_combined_names() {
  using measurements = metafunctions::measurements_t<ControlSystems>;
  using control_systems_with_measurements =
      tmpl::transform<measurements,
                      metafunctions::control_systems_with_measurement<
                          tmpl::pin<ControlSystems>, tmpl::_1>>;

  std::unordered_map<std::string, std::string> map_of_names{};

  tmpl::for_each<control_systems_with_measurements>([&map_of_names](
                                                        auto list_v) {
    using control_systems_with_measurement = tmpl::type_from<decltype(list_v)>;

    const std::string combined_name =
        control_system::combined_name<control_systems_with_measurement>();

    tmpl::for_each<control_systems_with_measurement>(
        [&map_of_names, &combined_name](auto control_system_v) {
          using control_system = tmpl::type_from<decltype(control_system_v)>;
          map_of_names[control_system::name()] = combined_name;
        });
  });

  return map_of_names;
}
}  // namespace control_system
