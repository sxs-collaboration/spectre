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
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace deadlock {
/*!
 * \brief Simple action to print information from the Interpolator.
 *
 * \details Makes a directory called `interpolator` inside the \p deadlock_dir
 * if it doesn't exist. Then writes to a new file for each target the following
 * information only for sequential targets for all temporal ids stored:
 *
 * - Interpolator core
 * - Temporal id
 * - Iteration number
 * - Expected number of elements to receive
 * - Current number of elements received
 * - Missing elements
 */
struct PrintInterpolator {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const std::string& deadlock_dir) {
    using target_tags = typename Metavariables::interpolation_target_tags;

    const std::string intrp_deadlock_dir = deadlock_dir + "/interpolator";
    if (not file_system::check_if_dir_exists(intrp_deadlock_dir)) {
      file_system::create_directory(intrp_deadlock_dir);
    }

    tmpl::for_each<target_tags>([&](const auto tag_v) {
      using target_tag = tmpl::type_from<decltype(tag_v)>;
      const std::string& target_name = pretty_type::name<target_tag>();

      // Only need to print the sequential targets (aka horizons)
      if constexpr (target_tag::compute_target_points::is_sequential::value) {
        const std::string file_name =
            intrp_deadlock_dir + "/" + target_name + ".out";

        const auto& holders =
            db::get<intrp::Tags::InterpolatedVarsHolders<Metavariables>>(box);
        const auto& vars_infos =
            get<intrp::Vars::HolderTag<target_tag, Metavariables>>(holders)
                .infos;
        const auto& num_elements =
            db::get<intrp::Tags::NumberOfElements<Metavariables::volume_dim>>(
                box);

        if (vars_infos.empty()) {
          Parallel::fprintf(file_name, "No data on interpolator core %zu\n",
                            array_index);
          return;
        }

        std::stringstream ss{};
        ss << std::setprecision(std::numeric_limits<double>::digits10 + 4)
           << std::scientific;

        ss << "========== BEGIN INTERPOLATOR CORE " << array_index
           << " ==========\n";

        for (const auto& [temporal_id, info] : vars_infos) {
          ss << "Temporal id " << temporal_id << ": "
             << "Iteration " << info.iteration << ", expecting data from "
             << num_elements.at(target_name).size()
             << " elements, but only received "
             << info.interpolation_is_done_for_these_elements.size()
             << ". Missing these elements: ";

          std::unordered_set<ElementId<Metavariables::volume_dim>> difference{};
          for (const auto& element : num_elements.at(target_name)) {
            if (not info.interpolation_is_done_for_these_elements.contains(
                    element)) {
              difference.insert(element);
            }
          }

          ss << difference << "\n";
        }

        ss << "========== END INTERPOLATOR CORE " << array_index
           << " ============\n";

        Parallel::fprintf(file_name, "%s\n", ss.str());
      }
    });

    (void)box;
    (void)cache;
    (void)array_index;
  }
};
}  // namespace deadlock
