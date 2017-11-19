// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.decl.h"
#include "Parallel/ConstGlobalCache.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace MainTestObjects {
struct CreatableNonCopyable {
  CreatableNonCopyable() = default;
  CreatableNonCopyable(const CreatableNonCopyable&) = delete;
  CreatableNonCopyable(CreatableNonCopyable&&) = default;
  CreatableNonCopyable& operator=(const CreatableNonCopyable&) = delete;
  CreatableNonCopyable& operator=(CreatableNonCopyable&&) = default;
  ~CreatableNonCopyable() = default;

  using options = tmpl::list<>;
  static constexpr OptionString help{""};

  // clang-tidy: google-runtime-references
  void pup(PUP::er& /*p*/) {}  // NOLINT
};

inline std::ostream& operator<<(std::ostream& s,
                                const CreatableNonCopyable& /*value*/) {
  return s;
}
}  // namespace MainTestObjects

namespace Test_Tentacles {

namespace Options {
// We can't actually parse an input file in the unit test, so these
// all have to have default values.
struct Integer {
  using type = int;
  static constexpr OptionString help{"halp"};
  static type default_value() { return 7; }
};

struct NonCopyable {
  using type = MainTestObjects::CreatableNonCopyable;
  static constexpr OptionString help{"halp"};
  static type default_value() { return {}; }
};
}  // namespace Options

template <typename Metavariables>
struct Group {
  using type = CProxy_TestGroupChare<Metavariables>;
  using const_global_cache_tag_list = typelist<Options::NonCopyable>;
  using options = typelist<Options::Integer>;
  static void initialize(
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/,
      const int& /*unused*/) {}
  static void execute_next_global_actions(
      const typename Metavariables::phase /*unused*/,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/) {}
};

template <typename Metavariables>
struct NodeGroup {
  using type = CProxy_TestNodeGroupChare<Metavariables>;
  using const_global_cache_tag_list = typelist<>;
  using options = typelist<Options::NonCopyable>;
  static void initialize(
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/,
      const MainTestObjects::CreatableNonCopyable& /*unused*/) {}
  static void execute_next_global_actions(
      const typename Metavariables::phase /*unused*/,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/) {}
};

template <typename Metavariables>
struct Chare {
  using type = CProxy_TestChare<Metavariables>;
  using const_global_cache_tag_list = typelist<>;
  using options = typelist<Options::Integer>;
  static void initialize(
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& cache_proxy,
      const int& integer) {
    Parallel::ConstGlobalCache<Metavariables>& cache =
        *cache_proxy.ckLocalBranch();
    auto& my_proxy = cache.template get_tentacle<Chare<Metavariables>>();
    my_proxy.set_id(integer);
  }
  static void execute_next_global_actions(
      const typename Metavariables::phase /*unused*/,
      const Parallel::CProxy_ConstGlobalCache<Metavariables>& /*unused*/) {}
};

template <typename Metavariables>
struct Array {
  using type = CProxy_TestArrayChare<Metavariables>;
  using const_global_cache_tag_list = typelist<>;
  using options = typelist<>;
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
  using const_global_cache_tag_list = typelist<>;
  using options = typelist<>;
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
