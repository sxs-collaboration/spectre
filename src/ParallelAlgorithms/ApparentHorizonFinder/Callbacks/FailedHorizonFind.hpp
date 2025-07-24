// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <set>
#include <sstream>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace ah::callbacks {
/*!
 * \brief Callback when an apparent horizon find fails. The template \p Ignore
 * says whether to ignore the failure or raise an ERROR.
 */
template <typename HorizonMetavars, bool Ignore>
struct FailedHorizonFind : tt::ConformsTo<ah::protocols::Callback> {
 private:
  using Fr = typename HorizonMetavars::frame;

 public:
  template <typename DbTags, typename Metavariables>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const FastFlow::Status failure_reason) {
    std::ostringstream os{};

    const auto& time = db::get<ah::Tags::CurrentTime>(box).value();
    const auto& all_storage = db::get<ah::Tags::Storage<Fr>>(box);
    const auto& current_time_storage = all_storage.at(time);
    const auto& current_iteration = current_time_storage.current_iteration;

    os << pretty_type::name<HorizonMetavars>()
       << ": Horizon find failed at time " << time
       << ". Reason = " << failure_reason
       << ". Number of compute coords retries = "
       << current_iteration.compute_coords_retries << ".";

    if (failure_reason == FastFlow::Status::InterpolationFailure) {
      const auto& block_coord_holders =
          current_iteration.block_coord_holders.value();

      std::vector<size_t> missing_indices{};
      missing_indices.reserve(block_coord_holders.size());
      for (size_t i = 0; i < block_coord_holders.size(); i++) {
        if (not block_coord_holders[i].has_value()) {
          missing_indices.push_back(i);
        }
      }

      // Get the actual points
      const auto& strahlkorper = current_iteration.strahlkorper;
      const auto& fast_flow = db::get<ah::Tags::FastFlow>(box);
      const size_t l_mesh = fast_flow.current_l_mesh(strahlkorper);
      const auto prolonged_strahlkorper =
          ylm::Strahlkorper<Fr>(l_mesh, l_mesh, strahlkorper);
      const auto coords = ylm::cartesian_coords(prolonged_strahlkorper);

      // Now output some information about them
      os << "\n Invalid points (in " << pretty_type::name<Fr>()
         << " frame) are:\n";
      for (const size_t index : missing_indices) {
        os << " (" << get<0>(coords)[index] << "," << get<1>(coords)[index]
           << "," << get<2>(coords)[index] << ")\n";
      }
    }

    if constexpr (Ignore) {
      const ::Verbosity verbosity = db::get<ah::Tags::Verbosity>(box);
      if (verbosity >= ::Verbosity::Quiet) {
        Parallel::printf("%s\n", os.str());
      }
    } else {
      ERROR(os.str());
    }
  }
};
}  // namespace ah::callbacks
