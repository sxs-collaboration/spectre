// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmGroup.hpp"
#include "AlgorithmNodegroup.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"

namespace observers {
/*!
 * \ingroup ObserversGroup
 * \brief The group parallel component that is responsible for reducing data
 * to be observed.
 *
 * Once the data from all elements on the processing element (usually a core)
 * has been collected, it is copied (not sent over the network) to the local
 * nodegroup parallel component, `ObserverWriter`, for writing to disk.
 */
template <class Metavariables>
struct Observer {
  using chare_type = Parallel::Algorithms::Group;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;

  using initial_databox = db::compute_databox_type<
      typename Actions::Initialize<Metavariables>::return_tag_list>;

  using options = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::simple_action<Actions::Initialize<Metavariables>>(
        Parallel::get_parallel_component<Observer>(local_cache));
  }

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) noexcept {}
};

/*!
 * \ingroup ObserversGroup
 * \brief The nodegroup parallel component that is responsible for writing data
 * to disk.
 */
template <class Metavariables>
struct ObserverWriter {
  using chare_type = Parallel::Algorithms::Nodegroup;
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::ReductionFileName, OptionTags::VolumeFileName>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;

  using initial_databox = db::compute_databox_type<
      typename Actions::InitializeWriter<Metavariables>::return_tag_list>;

  using options = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::simple_action<Actions::InitializeWriter<Metavariables>>(
        Parallel::get_parallel_component<ObserverWriter>(local_cache));
  }

  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) noexcept {}
};
}  // namespace observers
