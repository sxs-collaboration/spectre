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
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace deadlock {
/*!
 * \brief Simple action to print deadlock info on an interpolation target.
 *
 * \details This will print the following information for all temporal ids
 * stored in `intrp::Tags::TemporalIds`.
 *
 * - `intrp::Tags::IndicesOfFilledInterpPoints`
 * - `intrp::Tags::IndicesOfInvalidInterpPoints`
 * - Size of `intrp::Tags::InterpolatedVars`
 */
struct PrintInterpolationTarget {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const std::string& file_name) {
    std::stringstream ss{};
    ss << std::setprecision(std::numeric_limits<double>::digits10 + 4)
       << std::scientific;

    using TargetTag = typename ParallelComponent::interpolation_target_tag;
    using TemporalId = typename TargetTag::temporal_id::type;

    const auto stream_points = [&](const TemporalId& temporal_id) {
      const auto& filled_indices =
          db::get<intrp::Tags::IndicesOfFilledInterpPoints<TemporalId>>(box);
      const auto& invalid_indices =
          db::get<intrp::Tags::IndicesOfInvalidInterpPoints<TemporalId>>(box);
      const auto& interpolated_vars =
          db::get<intrp::Tags::InterpolatedVars<TargetTag, TemporalId>>(box);

      const size_t expected_size =
          interpolated_vars.at(temporal_id).number_of_grid_points();
      const size_t filled_size = filled_indices.count(temporal_id) > 0
                                     ? filled_indices.at(temporal_id).size()
                                     : 0_st;
      const size_t invalid_size = invalid_indices.count(temporal_id) > 0
                                      ? invalid_indices.at(temporal_id).size()
                                      : 0_st;

      ss << "Total points expected = " << expected_size
         << ", valid points received = " << filled_size
         << ", invalid points received " << invalid_size << ". ";
    };

    const auto& temporal_ids =
        db::get<intrp::Tags::TemporalIds<TemporalId>>(box);

    if (temporal_ids.empty()) {
      ss << pretty_type::name<TargetTag>() << ", No temporal ids.";
      Parallel::printf("%s\n", ss.str());
      return;
    }

    ss << "========== BEGIN TARGET " << pretty_type::name<TargetTag>()
       << " ==========\n";

    for (const auto& temporal_id : temporal_ids) {
      ss << "Temporal id " << temporal_id << ", ";

      stream_points(temporal_id);

      ss << "\n";
    }

    ss << "========== END TARGET " << pretty_type::name<TargetTag>()
       << " ============\n";

    Parallel::fprintf(file_name, "%s\n", ss.str());
  }
};
}  // namespace deadlock
