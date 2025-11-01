// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iomanip>
#include <ios>
#include <limits>
#include <sstream>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::Actions {
/*!
 * \brief Simple action to print deadlock info of the horizon finder.
 */
struct PrintDeadlockAnalysis {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const std::string& file_name) {
    std::stringstream ss{};
    ss << std::setprecision(std::numeric_limits<double>::digits10 + 4)
       << std::scientific;

    using HorizonMetavars = typename ParallelComponent::horizon_metavars;
    const std::string& name = pretty_type::name<HorizonMetavars>();
    ss << "Horizon finder " << name << ":\n";

    const auto& completed_times = db::get<ah::Tags::CompletedTimes>(box);
    ss << "  Completed times: " << completed_times << "\n";

    const auto& current_time_optional = db::get<ah::Tags::CurrentTime>(box);
    if (current_time_optional.has_value()) {
      ss << "  Current time: " << current_time_optional.value() << "\n";
    } else {
      ss << "  No current time set.\n";
    }

    const auto& pending_times = db::get<ah::Tags::PendingTimes>(box);
    ss << "  Pending times: " << pending_times << "\n";

    if (current_time_optional.has_value()) {
      const auto& current_time = current_time_optional.value();
      const auto& all_storage =
          db::get<ah::Tags::Storage<typename HorizonMetavars::frame>>(box);
      if (all_storage.contains(current_time)) {
        const auto& current_time_storage = all_storage.at(current_time);
        const auto& all_volume_variables =
            current_time_storage.all_volume_variables;
        const auto& current_iteration_storage =
            current_time_storage.current_iteration;
        const auto& fast_flow = db::get<ah::Tags::FastFlow>(box);

        ss << "  Time is ready: " << current_time_storage.time_is_ready << "\n";
        ss << "  FastFlow iteration: " << fast_flow.current_iteration() << "\n";
        ss << "  Coordinates are set: "
           << current_iteration_storage.block_coord_holders.has_value() << "\n";

        if (current_iteration_storage.block_coord_holders.has_value()) {
          ss << "  Number of points to interpolate to: "
             << current_iteration_storage.block_coord_holders->size() << "\n";
          const bool interpolation_is_complete =
              current_iteration_storage.interpolation_is_complete();
          ss << "  Interpolation is complete: " << interpolation_is_complete
             << "\n";
          if (not interpolation_is_complete) {
            ss << "  THE HORIZON FINDER IS STUCK IN INTERPOLATION, WHICH "
                  "LIKELY CAUSED THIS DEADLOCK. It is likely waiting for "
                  "volume data that will never arrive. See "
                  "'src/ParallelAlgorithms/ApparentHorizonFinder/Events/"
                  "FindApparentHorizon.hpp' for possible causes and "
                  "solutions.\n";
            const auto& block_logical_coords =
                current_iteration_storage.block_coord_holders.value();
            ss << "  Missing points (in block-logical coords):\n";
            const auto& filled =
                current_iteration_storage.indices_interpolated_to_thus_far;
            for (size_t i = 0; i < filled.size(); ++i) {
              if (not filled[i]) {
                ss << "    Index " << i << ": " << block_logical_coords[i]
                   << "\n";
              }
            }
          }
          ss << "  Intersecting element IDs: "
             << current_iteration_storage.intersecting_element_ids << "\n";
          ss << "  Element order: " << current_time_storage.element_order
             << "\n";
          ss << "  Volume data received from " << all_volume_variables.size()
             << " elements:\n";
          for (const auto& [element_id, volume_vars_storage] :
               all_volume_variables) {
            ss << "    " << element_id << "\n";
          }
        }
      } else {
        ss << "  No storage for current time " << current_time << ".\n";
      }
    }

    Parallel::fprintf(file_name, "%s\n", ss.str());
  }
};
}  // namespace ah::Actions
