// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template GlobalCache.

#pragma once

#include <charm++.h>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "Parallel/CharmRegistration.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

#include "Parallel/GlobalCache.decl.h"

namespace Parallel {

namespace GlobalCache_detail {

template <class GlobalCacheTag, class Metavariables>
using get_matching_tag = typename matching_tag_helper<
    GlobalCacheTag,
    tmpl::append<get_const_global_cache_tags<Metavariables>,
                 get_mutable_global_cache_tags<Metavariables>>>::type;

template <class GlobalCacheTag, class Metavariables>
using type_for_get = typename type_for_get_helper<
    typename get_matching_tag<GlobalCacheTag, Metavariables>::type>::type;

template <class T, class = std::void_t<>>
struct has_component_being_mocked_alias : std::false_type {};

template <class T>
struct has_component_being_mocked_alias<
    T, std::void_t<typename T::component_being_mocked>> : std::true_type {};

template <class T>
constexpr bool has_component_being_mocked_alias_v =
    has_component_being_mocked_alias<T>::value;

template <typename ComponentToFind, typename ComponentFromList>
struct get_component_if_mocked_helper {
  static_assert(
      has_component_being_mocked_alias_v<ComponentFromList>,
      "The parallel component was not found, and it looks like it is not being "
      "mocked. Did you forget to add it to the "
      "'Metavariables::component_list'? See the first template parameter for "
      "the component that we are looking for and the second template parameter "
      "for the component that is being checked for mocking it.");
  using type = std::is_same<typename ComponentFromList::component_being_mocked,
                            ComponentToFind>;
};

template <typename... Tags>
auto make_mutable_cache_tag_storage(
    tuples::TaggedTuple<Tags...>&& input) noexcept {
  return tuples::TaggedTuple<MutableCacheTag<Tags>...>(std::make_tuple(
      std::move(tuples::get<Tags>(input)), std::vector<CkCallback>{})...);
}

/// In order to be able to use a mock action testing framework we need to be
/// able to get the correct parallel component from the global cache even when
/// the correct component is a mock. We do this by having the mocked
/// components have a member type alias `component_being_mocked`, and having
/// `Parallel::get_component` check if the component to be retrieved is in the
/// `metavariables::component_list`. If it is not in the `component_list` then
/// we search for a mock component that is mocking the component we are trying
/// to retrieve.
template <typename ComponentList, typename ParallelComponent>
using get_component_if_mocked = tmpl::front<tmpl::type_from<tmpl::conditional_t<
    tmpl::list_contains_v<ComponentList, ParallelComponent>,
    tmpl::type_<tmpl::list<ParallelComponent>>,
    tmpl::lazy::find<ComponentList,
                     tmpl::type_<get_component_if_mocked_helper<
                         tmpl::pin<ParallelComponent>, tmpl::_1>>>>>>;
}  // namespace GlobalCache_detail

/// \ingroup ParallelGroup
/// A Charm++ chare that caches mutable data once per Charm++ core.
///
/// `MutableGlobalCache` is not intended to be visible to the end user; its
/// interface is via the `GlobalCache` member functions
/// `mutable_cache_item_is_ready`, `mutate`, and `get`.
/// Accordingly, most documentation of `MutableGlobalCache` is provided
/// in the relevant `GlobalCache` member functions.
template <typename Metavariables>
class MutableGlobalCache : public CBase_MutableGlobalCache<Metavariables> {
 public:
  explicit MutableGlobalCache(tuples::tagged_tuple_from_typelist<
                              get_mutable_global_cache_tags<Metavariables>>
                                  mutable_global_cache) noexcept;
  explicit MutableGlobalCache(CkMigrateMessage* msg)
      : CBase_MutableGlobalCache<Metavariables>(msg) {}

  ~MutableGlobalCache() noexcept override {
    (void)Parallel::charmxx::RegisterChare<
        MutableGlobalCache<Metavariables>,
        CkIndex_MutableGlobalCache<Metavariables>>::registrar;
  }
  /// \cond
  MutableGlobalCache() = default;
  MutableGlobalCache(const MutableGlobalCache&) = default;
  MutableGlobalCache& operator=(const MutableGlobalCache&) = default;
  MutableGlobalCache(MutableGlobalCache&&) = default;
  MutableGlobalCache& operator=(MutableGlobalCache&&) = default;
  /// \endcond

  template <typename GlobalCacheTag>
  auto get() const noexcept
      -> const GlobalCache_detail::type_for_get<GlobalCacheTag, Metavariables>&;

  // Entry method to mutate the object indentified by `GlobalCacheTag`.
  // Internally calls Function::apply(), where
  // Function is a struct, and Function::apply is a user-defined
  // static function that mutates the object.  Function::apply() takes
  // as its first argument a gsl::not_null pointer to the object named
  // by the GlobalCacheTag, and then the contents of 'args' as
  // subsequent arguments.  Called via `GlobalCache::mutate`.
  template <typename GlobalCacheTag, typename Function, typename... Args>
  void mutate(const std::tuple<Args...>& args) noexcept;

  // Not an entry method, and intended to be called only from
  // `GlobalCache` via the free function
  // `Parallel::mutable_cache_item_is_ready`.  See the free function
  // `Parallel::mutable_cache_item_is_ready` for documentation.
  template <typename GlobalCacheTag, typename Function>
  bool mutable_cache_item_is_ready(const Function& function) noexcept;

  void pup(PUP::er& p) noexcept override;  // NOLINT

 private:
  tuples::tagged_tuple_from_typelist<
      get_mutable_global_cache_tag_storage<Metavariables>>
      mutable_global_cache_{};
};

template <typename Metavariables>
MutableGlobalCache<Metavariables>::MutableGlobalCache(
    tuples::tagged_tuple_from_typelist<
        get_mutable_global_cache_tags<Metavariables>>
        mutable_global_cache) noexcept
    : mutable_global_cache_(GlobalCache_detail::make_mutable_cache_tag_storage(
          std::move(mutable_global_cache))) {}

template <typename Metavariables>
template <typename GlobalCacheTag>
auto MutableGlobalCache<Metavariables>::get() const noexcept
    -> const GlobalCache_detail::type_for_get<GlobalCacheTag, Metavariables>& {
  using tag = MutableCacheTag<
      GlobalCache_detail::get_matching_tag<GlobalCacheTag, Metavariables>>;
  if constexpr (tt::is_a_v<std::unique_ptr, typename tag::tag::type>) {
    return *(std::get<0>(tuples::get<tag>(mutable_global_cache_)));
  } else {
    return std::get<0>(tuples::get<tag>(mutable_global_cache_));
  }
}

template <typename Metavariables>
template <typename GlobalCacheTag, typename Function>
bool MutableGlobalCache<Metavariables>::mutable_cache_item_is_ready(
    const Function& function) noexcept {
  using tag = MutableCacheTag<GlobalCache_detail::get_matching_mutable_tag<
      GlobalCacheTag, Metavariables>>;
  std::optional<CkCallback> optional_callback{};
  if constexpr (tt::is_a_v<std::unique_ptr, typename tag::tag::type>) {
    optional_callback =
        function(*(std::get<0>(tuples::get<tag>(mutable_global_cache_))));
  } else {
    optional_callback =
        function(std::get<0>(tuples::get<tag>(mutable_global_cache_)));
  }
  if (optional_callback) {
    std::get<1>(tuples::get<tag>(mutable_global_cache_))
        .push_back(std::move(*optional_callback));
    if (std::get<1>(tuples::get<tag>(mutable_global_cache_)).size() > 20000) {
      ERROR("The number of callbacks in MutableGlobalCache for tag "
            << pretty_type::short_name<GlobalCacheTag>()
            << " has gotten too large, and may be growing without bound");
    }
    return false;
  } else {
    // The user-defined `function` didn't specify a callback, which
    // means that the item is ready.
    return true;
  }
}

template <typename Metavariables>
template <typename GlobalCacheTag, typename Function, typename... Args>
void MutableGlobalCache<Metavariables>::mutate(
    const std::tuple<Args...>& args) noexcept {
  (void)Parallel::charmxx::RegisterMutableGlobalCacheMutate<
      Metavariables, GlobalCacheTag, Function, Args...>::registrar;
  using tag = MutableCacheTag<GlobalCache_detail::get_matching_mutable_tag<
      GlobalCacheTag, Metavariables>>;

  // Do the mutate.
  std::apply(
      [this](const auto&... local_args) noexcept {
        Function::apply(make_not_null(&std::get<0>(
                            tuples::get<tag>(mutable_global_cache_))),
                        local_args...);
      },
      args);

  // Call the callbacks and clear the list of callbacks.
  for (auto& callback : std::get<1>(tuples::get<tag>(mutable_global_cache_))) {
    callback.send(nullptr);
  }
  std::get<1>(tuples::get<tag>(mutable_global_cache_)).clear();
  std::get<1>(tuples::get<tag>(mutable_global_cache_)).shrink_to_fit();
}

template <typename Metavariables>
void MutableGlobalCache<Metavariables>::pup(PUP::er& p) noexcept {
  p | mutable_global_cache_;
}

/// \ingroup ParallelGroup
/// A Charm++ chare that caches constant data once per Charm++ node or
/// non-constant data once per Charm++ core.
///
/// `Metavariables` must define the following metavariables:
///   - `component_list`   typelist of ParallelComponents
///   - `const_global_cache_tags`   (possibly empty) typelist of tags of
///     constant data
///   - `mutable_global_cache_tags` (possibly empty) typelist of tags of
///     non-constant data
///
/// The tag lists for the const items added to the GlobalCache is created by
/// combining the following tag lists:
///   - `Metavariables::const_global_cache_tags` which should contain only those
///     tags that cannot be added from the other tag lists below.
///   - `Component::const_global_cache_tags` for each `Component` in
///     `Metavariables::component_list` which should contain the tags needed by
///     any simple actions called on the Component, as well as tags need by the
///     `allocate_array` function of an array component.  The type alias may be
///     omitted for an empty list.
///   - `Action::const_global_cache_tags` for each `Action` in the
///     `phase_dependent_action_list` of each `Component` of
///     `Metavariables::component_list` which should contain the tags needed by
///     that  Action.  The type alias may be omitted for an empty list.
///
/// The tag lists for the non-const items added to the GlobalCache is created
/// by combining exactly the same tag lists as for the const items, except with
/// `const_global_cache_tags` replaced by `mutable_global_cache_tags`.
///
/// The tags in the `const_global_cache_tags` and
/// `mutable_global_cache_tags` type lists are db::SimpleTag%s that
/// have a `using option_tags` type alias and a static function
/// `create_from_options` that are used to create the constant data
/// from input file options.
///
/// References to const items in the GlobalCache are also added to the
/// db::DataBox of each `Component` in the
/// `Metavariables::component_list` with the same tag with which they
/// were inserted into the GlobalCache.  References to non-const items
/// in the GlobalCache are not added to the db::DataBox.
template <typename Metavariables>
class GlobalCache : public CBase_GlobalCache<Metavariables> {
  using parallel_component_tag_list = tmpl::transform<
      typename Metavariables::component_list,
      tmpl::bind<
          tmpl::type_,
          tmpl::bind<Parallel::proxy_from_parallel_component, tmpl::_1>>>;

 public:
  using proxy_type = CProxy_GlobalCache<Metavariables>;
  /// Access to the Metavariables template parameter
  using metavariables = Metavariables;
  /// Typelist of the ParallelComponents stored in the GlobalCache
  using component_list = typename Metavariables::component_list;

  /// Constructor used only by the ActionTesting framework and other
  /// non-charm++ tests that don't know about proxies.
  GlobalCache(tuples::tagged_tuple_from_typelist<
                  get_const_global_cache_tags<Metavariables>>
                  const_global_cache,
              MutableGlobalCache<Metavariables>* mutable_global_cache) noexcept;

  /// Constructor used by Main and anything else that is charm++ aware.
  GlobalCache(tuples::tagged_tuple_from_typelist<
                  get_const_global_cache_tags<Metavariables>>
                  const_global_cache,
              CProxy_MutableGlobalCache<Metavariables>
                  mutable_global_cache_proxy) noexcept;

  explicit GlobalCache(CkMigrateMessage* msg)
      : CBase_GlobalCache<Metavariables>(msg) {}

  ~GlobalCache() noexcept override {
    (void)Parallel::charmxx::RegisterChare<
        GlobalCache<Metavariables>,
        CkIndex_GlobalCache<Metavariables>>::registrar;
  }
  /// \cond
  GlobalCache() = default;
  GlobalCache(const GlobalCache&) = default;
  GlobalCache& operator=(const GlobalCache&) = default;
  GlobalCache(GlobalCache&&) = default;
  GlobalCache& operator=(GlobalCache&&) = default;
  /// \endcond

  /// Entry method to set the ParallelComponents (should only be called once)
  void set_parallel_components(
      tuples::tagged_tuple_from_typelist<parallel_component_tag_list>&&
          parallel_components,
      const CkCallback& callback) noexcept;

  /// Returns whether the object referred to by `GlobalCacheTag`
  /// (which must be a mutable cache tag) is ready to be accessed by a
  /// `get` call.
  ///
  /// `function` is a user-defined invokable that:
  /// - takes one argument: a const reference to the object referred to by the
  ///   `GlobalCacheTag`.
  /// - if the data is ready, returns a default constructed
  ///   `std::optional<CkCallBack>`
  /// - if the data is not ready, returns a `std::optional<CkCallBack>`,
  ///   where the `CkCallback` will re-invoke the current action on the
  ///   current parallel component.
  template <typename GlobalCacheTag, typename Function>
  bool mutable_cache_item_is_ready(const Function& function) noexcept;

  /// Mutates the non-const object identified by GlobalCacheTag.
  /// \requires `GlobalCacheTag` is a tag in `mutable_global_cache_tags`
  /// defined by the Metavariables and in Actions.
  ///
  /// Internally calls `Function::apply()`, where `Function` is a
  /// user-defined struct and `Function::apply()` is a user-defined
  /// static function that mutates the object.  `Function::apply()`
  /// takes as its first argument a `gsl::not_null` pointer to the
  /// object named by the GlobalCacheTag, and takes the contents of
  /// `args` as subsequent arguments.
  template <typename GlobalCacheTag, typename Function, typename... Args>
  void mutate(const std::tuple<Args...>& args) noexcept;

  /// Retrieve the proxy to the global cache
  proxy_type get_this_proxy() noexcept;

  void pup(PUP::er& p) noexcept override;  // NOLINT

 private:
  // clang-tidy: false positive, redundant declaration
  template <typename GlobalCacheTag, typename MV>
  friend auto get(const GlobalCache<MV>& cache) noexcept  // NOLINT
      -> const GlobalCache_detail::type_for_get<GlobalCacheTag, MV>&;

  // clang-tidy: false positive, redundant declaration
  template <typename ParallelComponentTag, typename MV>
  friend auto get_parallel_component(  // NOLINT
      GlobalCache<MV>& cache) noexcept
      -> Parallel::proxy_from_parallel_component<
          GlobalCache_detail::get_component_if_mocked<
              typename MV::component_list, ParallelComponentTag>>&;

  // clang-tidy: false positive, redundant declaration
  template <typename ParallelComponentTag, typename MV>
  friend auto get_parallel_component(  // NOLINT
      const GlobalCache<MV>& cache) noexcept
      -> const Parallel::proxy_from_parallel_component<
          GlobalCache_detail::get_component_if_mocked<
              typename MV::component_list,
              ParallelComponentTag>>&;  // NOLINT

  tuples::tagged_tuple_from_typelist<get_const_global_cache_tags<Metavariables>>
      const_global_cache_{};
  tuples::tagged_tuple_from_typelist<parallel_component_tag_list>
      parallel_components_{};
  // We store both a pointer and a proxy to the MutableGlobalCache.
  // There is both a pointer and a proxy because we want to use
  // MutableGlobalCache in production (where it must be charm-aware)
  // and for simple testing (which we want to do in a non-charm-aware
  // context for simplicity).  If the charm-aware constructor is
  // used, then the pointer is set to nullptr and the proxy is set.
  // If the non-charm-aware constructor is used, the the pointer is
  // set and the proxy is ignored.  The member functions that need the
  // MutableGlobalCache should use the pointer if it is not nullptr,
  // otherwise use the proxy.
  MutableGlobalCache<Metavariables>* mutable_global_cache_{nullptr};
  CProxy_MutableGlobalCache<Metavariables> mutable_global_cache_proxy_{};
  bool parallel_components_have_been_set_{false};
};

template <typename Metavariables>
GlobalCache<Metavariables>::GlobalCache(
    tuples::tagged_tuple_from_typelist<
        get_const_global_cache_tags<Metavariables>>
        const_global_cache,
    MutableGlobalCache<Metavariables>* mutable_global_cache) noexcept
    : const_global_cache_(std::move(const_global_cache)),
      mutable_global_cache_(mutable_global_cache) {
  ASSERT(mutable_global_cache_ != nullptr,
         "GlobalCache: Do not construct with a nullptr!");
}

template <typename Metavariables>
GlobalCache<Metavariables>::GlobalCache(
    tuples::tagged_tuple_from_typelist<
        get_const_global_cache_tags<Metavariables>>
        const_global_cache,
    CProxy_MutableGlobalCache<Metavariables>
        mutable_global_cache_proxy) noexcept
    : const_global_cache_(std::move(const_global_cache)),
      mutable_global_cache_(nullptr),
      mutable_global_cache_proxy_(std::move(mutable_global_cache_proxy)) {}

template <typename Metavariables>
void GlobalCache<Metavariables>::set_parallel_components(
    tuples::tagged_tuple_from_typelist<parallel_component_tag_list>&&
        parallel_components,
    const CkCallback& callback) noexcept {
  ASSERT(!parallel_components_have_been_set_,
         "Can only set the parallel_components once");
  parallel_components_ = std::move(parallel_components);
  parallel_components_have_been_set_ = true;
  this->contribute(callback);
}

template <typename Metavariables>
template <typename GlobalCacheTag, typename Function>
bool GlobalCache<Metavariables>::mutable_cache_item_is_ready(
    const Function& function) noexcept {
  if (mutable_global_cache_ == nullptr) {
    return mutable_global_cache_proxy_.ckLocalBranch()
        ->template mutable_cache_item_is_ready<GlobalCacheTag>(function);
  } else {
    return mutable_global_cache_
        ->template mutable_cache_item_is_ready<GlobalCacheTag>(function);
  }
}

template <typename Metavariables>
template <typename GlobalCacheTag, typename Function, typename... Args>
void GlobalCache<Metavariables>::mutate(
    const std::tuple<Args...>& args) noexcept {
  (void)Parallel::charmxx::RegisterGlobalCacheMutate<
      Metavariables, GlobalCacheTag, Function, Args...>::registrar;
  if (mutable_global_cache_ == nullptr) {
    // charm-aware version: Mutate the variable on all PEs on this node.
    for (auto pe = CkNodeFirst(CkMyNode());
         pe < CkNodeFirst(CkMyNode()) + CkNodeSize(CkMyNode()); ++pe) {
      mutable_global_cache_proxy_[pe].template mutate<GlobalCacheTag, Function>(
          args);
    }
  } else {
    // version that bypasses proxies.  Just call the function.
    mutable_global_cache_->template mutate<GlobalCacheTag, Function>(args);
  }
}

template <typename Metavariables>
typename Parallel::GlobalCache<Metavariables>::proxy_type
GlobalCache<Metavariables>::get_this_proxy() noexcept {
  return this->thisProxy;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)
template <typename Metavariables>
void GlobalCache<Metavariables>::pup(PUP::er& p) noexcept {
  p | const_global_cache_;
  p | parallel_components_;
  p | mutable_global_cache_proxy_;
  p | parallel_components_have_been_set_;
  if (not p.isUnpacking() and mutable_global_cache_ != nullptr) {
    ERROR(
        "Cannot serialize the const global cache when the mutable global cache "
        "is set to a local pointer. If this occurs in a unit test, avoid the "
        "serialization. If this occurs in a production executable, be sure "
        "that the MutableGlobalCache is accessed by a charm proxy.");
  }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)

// @{
/// \ingroup ParallelGroup
/// \brief Access the Charm++ proxy associated with a ParallelComponent
///
/// \requires ParallelComponentTag is a tag in component_list
///
/// \returns a Charm++ proxy that can be used to call an entry method on the
/// chare(s)
template <typename ParallelComponentTag, typename Metavariables>
auto get_parallel_component(GlobalCache<Metavariables>& cache) noexcept
    -> Parallel::proxy_from_parallel_component<
        GlobalCache_detail::get_component_if_mocked<
            typename Metavariables::component_list, ParallelComponentTag>>& {
  return tuples::get<tmpl::type_<Parallel::proxy_from_parallel_component<
      GlobalCache_detail::get_component_if_mocked<
          typename Metavariables::component_list, ParallelComponentTag>>>>(
      cache.parallel_components_);
}

template <typename ParallelComponentTag, typename Metavariables>
auto get_parallel_component(const GlobalCache<Metavariables>& cache) noexcept
    -> const Parallel::proxy_from_parallel_component<
        GlobalCache_detail::get_component_if_mocked<
            typename Metavariables::component_list, ParallelComponentTag>>& {
  return tuples::get<tmpl::type_<Parallel::proxy_from_parallel_component<
      GlobalCache_detail::get_component_if_mocked<
          typename Metavariables::component_list, ParallelComponentTag>>>>(
      cache.parallel_components_);
}
// @}

// @{
/// \ingroup ParallelGroup
/// \brief Access data in the cache
///
/// \requires GlobalCacheTag is a tag in the `mutable_global_cache_tags`
/// or `const_global_cache_tags` defined by the Metavariables and in Actions.
///
/// \returns a constant reference to an object in the cache
template <typename GlobalCacheTag, typename Metavariables>
auto get(const GlobalCache<Metavariables>& cache) noexcept
    -> const GlobalCache_detail::type_for_get<GlobalCacheTag, Metavariables>& {
  // We check if the tag is to be retrieved directly or via a base class
  using tag =
      GlobalCache_detail::get_matching_tag<GlobalCacheTag, Metavariables>;
  using tag_is_not_in_const_tags = std::is_same<
      tmpl::filter<get_const_global_cache_tags<Metavariables>,
                   std::is_base_of<tmpl::pin<GlobalCacheTag>, tmpl::_1>>,
      tmpl::list<>>;
  if constexpr (tag_is_not_in_const_tags::value) {
    // Tag is not in the const tags, so use MutableGlobalCache
    if (cache.mutable_global_cache_ == nullptr) {
      const auto& local_mutable_cache =
          *cache.mutable_global_cache_proxy_.ckLocalBranch();
      return local_mutable_cache.template get<GlobalCacheTag>();
    } else {
      return cache.mutable_global_cache_->template get<GlobalCacheTag>();
    }
  } else {
    // Tag is in the const tags, so use const_global_cache_
    if constexpr (tt::is_a_v<std::unique_ptr, typename tag::type>) {
      return *(tuples::get<tag>(cache.const_global_cache_));
    } else {
      return tuples::get<tag>(cache.const_global_cache_);
    }
  }
}

/// \ingroup ParallelGroup
/// \brief Returns whether the object identified by `GlobalCacheTag`
/// is ready to be accessed by `get`.
///
/// \requires `GlobalCacheTag` is a tag in `mutable_global_cache_tags`
/// defined by the Metavariables and in Actions.
///
/// \requires `function` is a user-defined invokable that takes one argument:
/// a const reference to the object referred to by the
/// `GlobalCacheTag`.  `function` returns a
/// `std::optional<CkCallBack>` that determines the readiness. To
/// indicate that the item is ready, the `std::optional` returned
/// by `function` must be invalid; in this case
/// `mutable_cache_item_is_ready` returns true. To indicate that the
/// item is not ready, the `std::optional` returned by `function`
/// must be valid; in this case, `mutable_cache_item_is_ready`
/// appends the `std::optional`'s wrapped `CkCallback` to an
/// internal list of callbacks to be called on `mutate`, and then
/// returns false.
template <typename GlobalCacheTag, typename Function, typename Metavariables>
bool mutable_cache_item_is_ready(GlobalCache<Metavariables>& cache,
                                 const Function& function) noexcept {
  return cache.template mutable_cache_item_is_ready<GlobalCacheTag>(function);
}

/// \ingroup ParallelGroup
/// \brief Mutates non-const data in the cache, by calling `Function::apply()`
///
/// \requires `GlobalCacheTag` is a tag in the `mutable_global_cache_tags`
/// defined by the Metavariables and in Actions.
/// \requires `Function` is a struct with a static void `apply()`
/// function that mutates the object. `Function::apply()` takes as its
/// first argument a `gsl::not_null` pointer to the object named by
/// the `GlobalCacheTag`, and takes `args` as
/// subsequent arguments.
///
/// This is the version that takes a GlobalCache<Metavariables>. Used only
/// for tests.
template <typename GlobalCacheTag, typename Function, typename Metavariables,
          typename... Args>
void mutate(GlobalCache<Metavariables>& cache, Args&&... args) noexcept {
  cache.template mutate<GlobalCacheTag, Function>(
      std::make_tuple<Args...>(std::forward<Args>(args)...));
}

/// \ingroup ParallelGroup
///
/// \brief Mutates non-const data in the cache, by calling `Function::apply()`
///
/// \requires `GlobalCacheTag` is a tag in tag_list.
/// \requires `Function` is a struct with a static void `apply()`
/// function that mutates the object. `Function::apply()` takes as its
/// first argument a `gsl::not_null` pointer to the object named by
/// the `GlobalCacheTag`, and takes `args` as
/// subsequent arguments.
///
/// This is the version that takes a charm++ proxy to the GlobalCache.
template <typename GlobalCacheTag, typename Function, typename Metavariables,
          typename... Args>
void mutate(CProxy_GlobalCache<Metavariables>& cache_proxy,
            Args&&... args) noexcept {
  cache_proxy.template mutate<GlobalCacheTag, Function>(
      std::make_tuple<Args...>(std::forward<Args>(args)...));
}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag to retrieve the `Parallel::GlobalCache` from the DataBox.
struct GlobalCache : db::BaseTag {};

template <class Metavariables>
struct GlobalCacheImpl : GlobalCache, db::SimpleTag {
  using type = Parallel::GlobalCache<Metavariables>*;
  static std::string name() noexcept { return "GlobalCache"; }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag used to retrieve data from the `Parallel::GlobalCache`. This is the
/// recommended way for compute tags to retrieve data out of the global cache.
template <class CacheTag>
struct FromGlobalCache : CacheTag, db::ComputeTag {
  static_assert(db::is_simple_tag_v<CacheTag>);
  static std::string name() noexcept {
    return "FromGlobalCache(" + pretty_type::short_name<CacheTag>() + ")";
  }
  template <class Metavariables>
  static const GlobalCache_detail::type_for_get<CacheTag, Metavariables>&
  function(const Parallel::GlobalCache<Metavariables>* const& cache) {
    return Parallel::get<CacheTag>(*cache);
  }
  using argument_tags = tmpl::list<GlobalCache>;
};
}  // namespace Tags
}  // namespace Parallel

namespace PUP {

/// \ingroup ParallelGroup
/// Serialization of a pointer to the global cache for Charm++
template <typename Metavariables>
inline void pup(PUP::er& p,  // NOLINT
                Parallel::GlobalCache<Metavariables>*& t) noexcept {
  typename Parallel::GlobalCache<Metavariables>::proxy_type
      local_const_global_cache_proxy;
  if (p.isUnpacking()) {
    p | local_const_global_cache_proxy;
    t = local_const_global_cache_proxy.ckLocalBranch();
  } else {
    local_const_global_cache_proxy = t->get_this_proxy();
    p | local_const_global_cache_proxy;
  }
}

/// \ingroup ParallelGroup
/// Serialization of a pointer to the global cache for Charm++
template <typename Metavariables>
inline void operator|(PUP::er& p,  // NOLINT
                      Parallel::GlobalCache<Metavariables>*& t) {
  pup(p, t);
}
}  // namespace PUP

#define CK_TEMPLATES_ONLY
#include "Parallel/GlobalCache.def.h"
#undef CK_TEMPLATES_ONLY
