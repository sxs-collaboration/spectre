// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "ApparentHorizons/Tags.hpp"
#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Tags/IsActive.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "Domain/Creators/Tags/ObjectCenter.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system {
namespace Actions {
namespace detail {
template <typename Objects>
struct get_center_tags;

template <domain::ObjectLabel... Object>
struct get_center_tags<domain::object_list<Object...>> {
  using type = tmpl::list<domain::Tags::ExcisionCenter<Object>...>;
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
 *   - `domain::Tags::ExcisionCenter<domain::ObjectLabel::A>`
 *   - `domain::Tags::ExcisionCenter<domain::ObjectLabel::B>`
 *
 * DataBox:
 * - Uses: Nothing
 * - Adds:
 *   - `control_system::Tags::Averager<ControlSystem>`
 *   - `control_system::Tags::Controller<ControlSystem>`
 *   - `control_system::Tags::TimescaleTuner<ControlSystem>`
 *   - `control_system::Tags::ControlError<ControlSystem>`
 *   - `control_system::Tags::IsActive<ControlSystem>`
 * - Removes: Nothing
 * - Modifies:
 *   - `control_system::Tags::Averager<ControlSystem>`
 *   - `control_system::Tags::CurrentNumberOfMeasurements`
 */
template <typename Metavariables, typename ControlSystem>
struct Initialize {
  static constexpr size_t deriv_order = ControlSystem::deriv_order;

  using simple_tags_from_options =
      tmpl::list<control_system::Tags::Averager<ControlSystem>,
                 control_system::Tags::Controller<ControlSystem>,
                 control_system::Tags::TimescaleTuner<ControlSystem>,
                 control_system::Tags::ControlError<ControlSystem>,
                 control_system::Tags::IsActive<ControlSystem>>;

  using simple_tags =
      tmpl::push_back<typename ControlSystem::simple_tags,
                      control_system::Tags::CurrentNumberOfMeasurements>;

  using const_global_cache_tags = tmpl::flatten<tmpl::list<
      control_system::Tags::MeasurementsPerUpdate,
      control_system::Tags::WriteDataToDisk,
      control_system::Tags::ObserveCenters, control_system::Tags::Verbosity,
      typename detail::get_center_tags<
          typename ControlSystem::control_error::object_centers>::type>>;

  using mutable_global_cache_tags =
      tmpl::list<control_system::Tags::MeasurementTimescales>;

  using compute_tags = tmpl::list<>;

  using return_tags =
      tmpl::list<control_system::Tags::Averager<ControlSystem>,
                 control_system::Tags::CurrentNumberOfMeasurements>;

  using argument_tags = tmpl::list<Parallel::Tags::GlobalCache>;

  static void apply(const gsl::not_null<::Averager<deriv_order - 1>*> averager,
                    const gsl::not_null<int*> current_number_of_measurements,
                    const Parallel::GlobalCache<Metavariables>* const& cache) {
    const auto& measurement_timescales =
        Parallel::get<control_system::Tags::MeasurementTimescales>(*cache);
    const auto& measurement_timescale_func =
        *(measurement_timescales.at(ControlSystem::name()));
    const double initial_time = measurement_timescale_func.time_bounds()[0];
    const double measurement_timescale =
        min(measurement_timescale_func.func(initial_time)[0]);

    averager->assign_time_between_measurements(measurement_timescale);
    *current_number_of_measurements = 0;
  }
};
}  // namespace Actions
}  // namespace control_system
