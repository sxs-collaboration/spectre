// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionCriteria.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Bbh/CompletionSingleton.hpp"
#include "NumericalAlgorithms/Strahlkorper/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::bbh::callbacks {
/*!
 * \brief Apparent-horizon callback that updates binary-black-hole completion
 * criteria from successful common-horizon finds.
 *
 * \details This callback forwards each successful AhC find (`temporal_id`,
 * `LMax`) to `gh::bbh::CompletionSingleton`, where completion criteria are
 * evaluated in a temporally robust, centralized way.
 */
template <typename HorizonMetavars>
struct UpdateCompletionCriteria : tt::ConformsTo<ah::protocols::Callback> {
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;

  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const FastFlow::Status /*status*/) {
    const auto& current_time = db::get<ah::Tags::CurrentTime>(box);
    ASSERT(current_time.has_value(),
           "AhC completion callback requires a current temporal id.");
    const size_t l_max =
        db::get<ylm::Tags::Strahlkorper<typename HorizonMetavars::frame>>(box)
            .l_max();
    Parallel::simple_action<gh::bbh::Actions::RecordCommonHorizonSuccess>(
        Parallel::get_parallel_component<
            gh::bbh::CompletionSingleton<Metavariables>>(cache),
        *current_time, l_max);
  }
};
}  // namespace gh::bbh::callbacks
