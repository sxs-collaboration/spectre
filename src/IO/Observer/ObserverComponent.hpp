// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmNodegroup.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"

namespace observers {
/*!
 * \ingroup ObserversGroup
 * \brief The nodegroup parallel component that is responsible for reducing data
 * to be observed and for writing data to disk.
 */
template <class Metavariables>
struct Observer {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;

  using initial_databox =
      db::compute_databox_type<typename Actions::Initialize::return_tag_list>;

  using options = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::simple_action<Actions::Initialize>(
        Parallel::get_parallel_component<Observer>(local_cache));
  }

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) noexcept {}
};

}  // namespace observers
