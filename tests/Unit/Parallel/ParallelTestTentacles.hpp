// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/ConstGlobalCache.decl.h"
#include "Parallel/ConstGlobalCache.hpp"

namespace Test_Tentacles {

template <typename Metavariables>
struct Group {
  using type = CProxy_TestGroupChare<Metavariables>;
  static void initialize(
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/) {}
  static void execute_next_global_actions(
      const typename Metavariables::phase /*unused*/,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/) {}
};

template <typename Metavariables>
struct NodeGroup {
  using type = CProxy_TestNodeGroupChare<Metavariables>;
  static void initialize(
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/) {}
  static void execute_next_global_actions(
      const typename Metavariables::phase /*unused*/,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/) {}
};

template <typename Metavariables>
struct Chare {
  using type = CProxy_TestChare<Metavariables>;
  static void initialize(
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& cache_proxy) {
    Parallel::ConstGlobalCache<Metavariables>& cache =
        *cache_proxy.ckLocalBranch();
    auto& my_proxy = cache.template get_tentacle<Chare<Metavariables>>();
    my_proxy.set_id(-1);
  }
  static void execute_next_global_actions(
      const typename Metavariables::phase /*unused*/,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/) {}
};

template <typename Metavariables>
struct Array {
  using type = CProxy_TestArrayChare<Metavariables>;
  static void initialize(
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& cache_proxy);
  static void execute_next_global_actions(
      const typename Metavariables::phase /*unused*/,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/) {}
};

template <typename Metavariables>
struct BoundArray {
  using type = CProxy_TestBoundArrayChare<Metavariables>;
  using bind_to = Test_Tentacles::Array<Metavariables>;
  static void initialize(
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/) {}
  static void execute_next_global_actions(
      const typename Metavariables::phase /*unused*/,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/) {}
};

template <typename Metavariables>
void Array<Metavariables>::initialize(
    const Parallel::CProxy_ConstGlobalCache<Metavariables>& cache_proxy) {
  Parallel::ConstGlobalCache<Metavariables>& cache =
      *cache_proxy.ckLocalBranch();
  auto& my_proxy = cache.template get_tentacle<Array<Metavariables>>();
  auto& bound_proxy = cache.template get_tentacle<BoundArray<Metavariables>>();
  for (int i = 0; i < 40; ++i) {
    my_proxy[i].insert(cache_proxy);
    bound_proxy[i].insert(cache_proxy);
  }
  my_proxy.doneInserting();
  bound_proxy.doneInserting();
}

}  // namespace Test_Tentacles
