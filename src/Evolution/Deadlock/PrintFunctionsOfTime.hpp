// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/FunctionsOfTime/OutputTimeBounds.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Printf/Printf.hpp"

namespace control_system::Tags {
struct MeasurementTimescales;
}  // namespace control_system::Tags

namespace deadlock {
/*!
 * \brief Simple action that will print the `domain::Tags::FunctionsOfTime` and
 * `control_system::Tags::MeasurementTimescales` (if it exists) time bounds for
 * each node of a simulation.
 */
struct PrintFunctionsOfTime {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const std::string& file_name) {
    const auto& functions_of_time =
        Parallel::get<::domain::Tags::FunctionsOfTime>(cache);
    const std::string time_bounds =
        ::domain::FunctionsOfTime::output_time_bounds(functions_of_time);

    if constexpr (Parallel::is_in_global_cache<
                      Metavariables,
                      control_system::Tags::MeasurementTimescales>) {
      const auto& measurement_timescales =
          Parallel::get<control_system::Tags::MeasurementTimescales>(cache);
      const std::string measurement_time_bounds =
          ::domain::FunctionsOfTime::output_time_bounds(measurement_timescales);

      Parallel::fprintf(
          file_name,
          "Node %i\nFunctionsOfTime:\n%s\n\nMeasurementTimescales:\n%s\n",
          Parallel::my_node(cache), time_bounds, measurement_time_bounds);
    } else {
      Parallel::fprintf(file_name, "Node %i\nFunctionsOfTime:%s\n",
                        Parallel::my_node(cache), time_bounds);
    }
  }
};
}  // namespace deadlock
