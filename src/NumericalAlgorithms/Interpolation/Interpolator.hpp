// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {

/// \brief ParallelComponent responsible for collecting data from
/// `Element`s and interpolating it onto `InterpolationTarget`s.
///
/// For requirements on Metavariables, see InterpolationTarget
template <class Metavariables>
struct Interpolator {
  using chare_type = Parallel::Algorithms::Group;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox =
      db::compute_databox_type<typename Actions::InitializeInterpolator::
                                   template return_tag_list<Metavariables>>;
  using options = tmpl::list<>;
  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache);
  static void execute_next_phase(typename Metavariables::Phase /*next_phase*/,
                                 const Parallel::CProxy_ConstGlobalCache<
                                     Metavariables>& /*global_cache*/){};
};

template <class Metavariables>
void Interpolator<Metavariables>::initialize(
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
  auto& my_proxy = Parallel::get_parallel_component<Interpolator>(
      *(global_cache.ckLocalBranch()));
  Parallel::simple_action<Actions::InitializeInterpolator>(my_proxy);
}

}  // namespace intrp
