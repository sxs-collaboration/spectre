// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Tags/IsActiveMap.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/UpdateFunctionOfTime.hpp"
#include "Domain/Creators/Tags/ObjectCenter.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system {
namespace Actions {
namespace detail {
template <typename Objects>
struct get_center_tags;

template <domain::ObjectLabel... Object>
struct get_center_tags<domain::object_list<Object...>> {
  using type = tmpl::list<domain::Tags::ObjectCenter<Object>...>;
};

template <>
struct get_center_tags<domain::object_list<>> {
  using type = tmpl::list<>;
};
}  // namespace detail

/*!
 * \ingroup ControlSystemGroup
 * \ingroup InitializationGroup
 * \brief Initialize items related to the control system
 *
 * GlobalCache:
 * - Uses:
 *   - `control_system::Tags::MeasurementTimescales`
 *   - `control_system::Tags::WriteDataToDisk`
 *   - `control_system::Tags::ObserveCenters`
 *   - `control_system::Tags::Verbosity`
 *   - `control_system::Tags::IsActiveMap`
 *   - `control_system::Tags::SystemToCombinedNames`
 *   - `domain::Tags::ObjectCenter<domain::ObjectLabel::A>`
 *   - `domain::Tags::ObjectCenter<domain::ObjectLabel::B>`
 *
 * DataBox:
 * - Uses: Nothing
 * - Adds:
 *   - `control_system::Tags::Averager<ControlSystem>`
 *   - `control_system::Tags::Controller<ControlSystem>`
 *   - `control_system::Tags::TimescaleTuner<ControlSystem>`
 *   - `control_system::Tags::ControlError<ControlSystem>`
 *   - `control_system::Tags::CurrentNumberOfMeasurements`
 *   - `control_system::Tags::UpdateAggregators`
 * - Removes: Nothing
 * - Modifies:
 *   - `control_system::Tags::Averager<ControlSystem>`
 *   - `control_system::Tags::CurrentNumberOfMeasurements`
 *   - `control_system::Tags::UpdateAggregators`
 */
template <typename Metavariables, typename ControlSystem>
struct Initialize {
  static constexpr size_t deriv_order = ControlSystem::deriv_order;

  using simple_tags_from_options =
      tmpl::list<control_system::Tags::Averager<ControlSystem>,
                 control_system::Tags::Controller<ControlSystem>,
                 control_system::Tags::TimescaleTuner<ControlSystem>,
                 control_system::Tags::ControlError<ControlSystem>>;

  using simple_tags =
      tmpl::push_back<typename ControlSystem::simple_tags,
                      control_system::Tags::UpdateAggregators,
                      control_system::Tags::CurrentNumberOfMeasurements>;

  using const_global_cache_tags = tmpl::flatten<tmpl::list<
      control_system::Tags::SystemToCombinedNames,
      control_system::Tags::MeasurementsPerUpdate,
      control_system::Tags::WriteDataToDisk,
      control_system::Tags::ObserveCenters, control_system::Tags::Verbosity,
      control_system::Tags::IsActiveMap,
      typename detail::get_center_tags<
          typename ControlSystem::control_error::object_centers>::type>>;

  using mutable_global_cache_tags =
      tmpl::list<control_system::Tags::MeasurementTimescales>;

  using compute_tags = tmpl::list<>;

  using return_tags =
      tmpl::list<control_system::Tags::Averager<ControlSystem>,
                 control_system::Tags::CurrentNumberOfMeasurements,
                 control_system::Tags::UpdateAggregators>;

  using argument_tags = tmpl::list<Parallel::Tags::GlobalCache>;

  static void apply(
      const gsl::not_null<::Averager<deriv_order - 1>*> averager,
      const gsl::not_null<int*> current_number_of_measurements,
      const gsl::not_null<
          std::unordered_map<std::string, control_system::UpdateAggregator>*>
          update_aggregators,
      const Parallel::GlobalCache<Metavariables>* const& cache) {
    const auto& measurement_timescales =
        Parallel::get<control_system::Tags::MeasurementTimescales>(*cache);
    const std::unordered_map<std::string, std::string>&
        system_to_combined_names =
            Parallel::get<control_system::Tags::SystemToCombinedNames>(*cache);
    const auto& measurement_timescale_func = *(measurement_timescales.at(
        system_to_combined_names.at(ControlSystem::name())));
    const double initial_time = measurement_timescale_func.time_bounds()[0];
    const double measurement_timescale =
        min(measurement_timescale_func.func(initial_time)[0]);

    averager->assign_time_between_measurements(measurement_timescale);
    *current_number_of_measurements = 0;

    const std::unordered_map<std::string, bool>& is_active_map =
        Parallel::get<control_system::Tags::IsActiveMap>(*cache);

    std::unordered_map<std::string, std::unordered_set<std::string>>
        combined_to_system_names{};
    for (const auto& [control_system_name, combined_name] :
         system_to_combined_names) {
      if (combined_to_system_names.count(combined_name) != 1) {
        combined_to_system_names[combined_name];
      }

      // Only add active control systems
      if (is_active_map.at(control_system_name)) {
        combined_to_system_names.at(combined_name).insert(control_system_name);
      }
    }

    for (auto& [combined_name, control_system_names] :
         combined_to_system_names) {
      (*update_aggregators)[combined_name] = control_system::UpdateAggregator{
          combined_name, std::move(control_system_names)};
    }
  }
};
}  // namespace Actions
}  // namespace control_system
