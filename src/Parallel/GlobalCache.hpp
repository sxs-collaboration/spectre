// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template GlobalCache.

#pragma once

#include <charm++.h>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/Callback.hpp"
#include "Parallel/CharmRegistration.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/ResourceInfo.hpp"
#include "Parallel/Tags/ResourceInfo.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

#include "Parallel/GlobalCache.decl.h"
#include "Parallel/Main.decl.h"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
namespace mem_monitor {
template <typename Metavariables>
struct MemoryMonitor;
template <typename ContributingComponent>
struct ContributeMemoryData;
}  // namespace mem_monitor
/// \endcond

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

CREATE_GET_TYPE_ALIAS_OR_DEFAULT(component_being_mocked)

template <typename... Tags>
auto make_mutable_cache_tag_storage(tuples::TaggedTuple<Tags...>&& input) {
  return tuples::TaggedTuple<MutableCacheTag<Tags>...>(
      std::make_tuple(std::move(tuples::get<Tags>(input)),
                      std::unordered_map<Parallel::ArrayComponentId,
                                         std::unique_ptr<Callback>>{})...);
}

template <typename ParallelComponent, typename ComponentList>
auto get_component_if_mocked_impl() {
  if constexpr (tmpl::list_contains_v<ComponentList, ParallelComponent>) {
    return ParallelComponent{};
  } else {
    using mock_find = tmpl::find<
        ComponentList,
        std::is_same<get_component_being_mocked_or_default<tmpl::_1, void>,
                     tmpl::pin<ParallelComponent>>>;
    static_assert(
        tmpl::size<mock_find>::value > 0,
        "The requested parallel component (first template argument) is not a "
        "known parallel component (second template argument) or a component "
        "being mocked by one of those components.");
    return tmpl::front<mock_find>{};
  }
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
using get_component_if_mocked =
    decltype(get_component_if_mocked_impl<ParallelComponent, ComponentList>());

/// This class replaces the Charm++ base class of the GlobalCache in unit tests
/// that don't have a Charm++ main function.
struct MockGlobalCache {
  MockGlobalCache() = default;
  explicit MockGlobalCache(CkMigrateMessage* /*msg*/) {}
  virtual ~MockGlobalCache() = default;
  virtual void pup(PUP::er& /*p*/) {}
};

#ifdef SPECTRE_CHARM_HAS_MAIN
static constexpr bool mock_global_cache = false;
#else
static constexpr bool mock_global_cache = true;
#endif  // SPECTRE_CHARM_HAS_MAIN

}  // namespace GlobalCache_detail

/// \cond
template <typename ParallelComponentTag, typename Metavariables>
auto get_parallel_component(GlobalCache<Metavariables>& cache)
    -> Parallel::proxy_from_parallel_component<
        GlobalCache_detail::get_component_if_mocked<
            typename Metavariables::component_list, ParallelComponentTag>>&;

template <typename ParallelComponentTag, typename Metavariables>
auto get_parallel_component(const GlobalCache<Metavariables>& cache)
    -> const Parallel::proxy_from_parallel_component<
        GlobalCache_detail::get_component_if_mocked<
            typename Metavariables::component_list, ParallelComponentTag>>&;
/// \endcond

/*!
 * \ingroup ParallelGroup
 * \brief A Charm++ chare that caches global data once per Charm++ node.
 *
 * \details There are two types of global data that are stored; const data and
 * mutable data. Once the GlobalCache is created, const data cannot be edited
 * but mutable data can be edited using `Parallel::mutate`.
 *
 * The template parameter `Metavariables` must define the following type
 * aliases:
 *   - `component_list`   typelist of ParallelComponents
 *   - `const_global_cache_tags`   (possibly empty) typelist of tags of
 *     constant data
 *   - `mutable_global_cache_tags` (possibly empty) typelist of tags of
 *     non-constant data
 *
 * The tag lists for the const items added to the GlobalCache is created by
 * combining the following tag lists:
 *   - `Metavariables::const_global_cache_tags` which should contain only those
 *     tags that cannot be added from the other tag lists below.
 *   - `Component::const_global_cache_tags` for each `Component` in
 *     `Metavariables::component_list` which should contain the tags needed by
 *     any simple actions called on the Component, as well as tags need by the
 *     `allocate_array` function of an array component.  The type alias may be
 *     omitted for an empty list.
 *   - `Action::const_global_cache_tags` for each `Action` in the
 *     `phase_dependent_action_list` of each `Component` of
 *     `Metavariables::component_list` which should contain the tags needed by
 *     that  Action.  The type alias may be omitted for an empty list.
 *
 * The tag lists for the mutable items added to the GlobalCache is created
 * by combining exactly the same tag lists as for the const items, except with
 * `const_global_cache_tags` replaced by `mutable_global_cache_tags`.
 *
 * The tags in the `const_global_cache_tags` and
 * `mutable_global_cache_tags` type lists are db::SimpleTag%s that
 * have a `using option_tags` type alias and a static function
 * `create_from_options` that are used to create the constant data (or initial
 * mutable data) from input file options.
 *
 * References to const items in the GlobalCache are also added to the
 * db::DataBox of each `Component` in the
 * `Metavariables::component_list` with the same tag with which they
 * were inserted into the GlobalCache.  References to mutable items
 * in the GlobalCache are not added to the db::DataBox.
 *
 * Since mutable data is stored once per Charm++ node, we require that
 * data structures held by mutable tags have some sort of thread-safety.
 * Particularly, we require data structures in mutable tags be Single
 * Producer-Multiple Consumer. This means that the data structure should be
 * readable/accessible by multiple threads at once, even while being mutated
 * (multiple consumer), but will not be edited/mutated simultaneously on
 * multiple threads (single producer).
 */
template <typename Metavariables>
class GlobalCache
    : public std::conditional_t<GlobalCache_detail::mock_global_cache,
                                GlobalCache_detail::MockGlobalCache,
                                CBase_GlobalCache<Metavariables>> {
  using Base = std::conditional_t<GlobalCache_detail::mock_global_cache,
                                  GlobalCache_detail::MockGlobalCache,
                                  CBase_GlobalCache<Metavariables>>;
  using parallel_component_tag_list = tmpl::transform<
      typename Metavariables::component_list,
      tmpl::bind<
          tmpl::type_,
          tmpl::bind<Parallel::proxy_from_parallel_component, tmpl::_1>>>;
  using ParallelComponentTuple =
      tuples::tagged_tuple_from_typelist<parallel_component_tag_list>;

 public:
  static constexpr bool is_mocked = GlobalCache_detail::mock_global_cache;
  using proxy_type = CProxy_GlobalCache<Metavariables>;
  using main_proxy_type = CProxy_Main<Metavariables>;
  /// Access to the Metavariables template parameter
  using metavariables = Metavariables;
  /// Typelist of the ParallelComponents stored in the GlobalCache
  using component_list = typename Metavariables::component_list;
  // Even though the GlobalCache doesn't run the Algorithm, this type alias
  // helps in identifying that the GlobalCache is a Nodegroup using
  // Parallel::is_nodegroup_v
  using chare_type = Parallel::Algorithms::Nodegroup;
  using const_tags_list = get_const_global_cache_tags<Metavariables>;
  using ConstTagsTuple = tuples::tagged_tuple_from_typelist<const_tags_list>;
  using ConstTagsStorage = ConstTagsTuple;
  using mutable_tags_list = get_mutable_global_cache_tags<Metavariables>;
  using MutableTagsTuple =
      tuples::tagged_tuple_from_typelist<mutable_tags_list>;
  using MutableTagsStorage = tuples::tagged_tuple_from_typelist<
      get_mutable_global_cache_tag_storage<Metavariables>>;

  /// Constructor meant to be used in the ActionTesting framework.
  GlobalCache(ConstTagsTuple const_global_cache,
              MutableTagsTuple mutable_global_cache = {},
              std::vector<size_t> procs_per_node = {1}, const int my_proc = 0,
              const int my_node = 0, const int my_local_rank = 0);

  /// Constructor meant to be used in charm-aware settings (with a Main proxy).
  GlobalCache(ConstTagsTuple const_global_cache,
              MutableTagsTuple mutable_global_cache,
              std::optional<main_proxy_type> main_proxy);

  explicit GlobalCache(CkMigrateMessage* msg) : Base(msg) {}

  ~GlobalCache() override {
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
  void set_parallel_components(ParallelComponentTuple&& parallel_components,
                               const CkCallback& callback);

  /*!
   * \brief Returns whether the object referred to by `GlobalCacheTag`
   * (which must be a mutable cache tag) is ready to be accessed by a
   * `get` call.
   *
   * \details `function` is a user-defined invokable that:
   * - takes one argument: a const reference to the object referred to by the
   *   `GlobalCacheTag`.
   * - if the data is ready, returns a default constructed
   *   `std::unique_ptr<CallBack>`
   * - if the data is not ready, returns a `std::unique_ptr<CallBack>`,
   *   where the `Callback` will re-invoke the current action on the
   *   current parallel component. This callback should be a
   *   `Parallel::PerformAlgorithmCallback`. Other types of callbacks are not
   *   supported at this time.
   *
   * \parblock
   * \warning The `function` may be called twice so it should not modify any
   * state in its scope.
   * \endparblock
   *
   * \parblock
   * \warning If there has already been a callback registered for the given
   * `array_component_id`, then the callback returned by `function` will **not**
   * be registered or called.
   * \endparblock
   */
  template <typename GlobalCacheTag, typename Function>
  bool mutable_cache_item_is_ready(
      const Parallel::ArrayComponentId& array_component_id,
      const Function& function);

  /// Mutates the non-const object identified by GlobalCacheTag.
  /// \requires `GlobalCacheTag` is a tag in `mutable_global_cache_tags`
  /// defined by the Metavariables and in Actions.
  ///
  /// Internally calls `Function::apply()`, where `Function` is a
  /// user-defined struct and `Function::apply()` is a user-defined
  /// static function that mutates the object.  `Function::apply()`
  /// takes as its first argument a `gsl::not_null` pointer to the
  /// object named by the GlobalCacheTag (or if that object is a
  /// `std::unique_ptr<T>`, a `gsl::not_null<T*>`), and takes the contents of
  /// `args` as subsequent arguments.
  template <typename GlobalCacheTag, typename Function, typename... Args>
  void mutate(const std::tuple<Args...>& args);

  /// Entry method that computes the size of the local branch of the
  /// GlobalCache and sends it to the MemoryMonitor parallel component.
  ///
  /// \note This can only be called if the MemoryMonitor component is in the
  /// `component_list` of the metavariables. Also can't be called in the testing
  /// framework. Trying to do either of these will result in an ERROR.
  void compute_size_for_memory_monitor(const double time);

  /// Entry method that will set the value of the Parallel::Tags::ResourceInfo
  /// tag to the value passed in (if the tag exists in the GlobalCache)
  ///
  /// This is only meant to be called once.
  void set_resource_info(
      const Parallel::ResourceInfo<Metavariables>& resource_info);

  /// Retrieve the resource_info
  const Parallel::ResourceInfo<Metavariables>& get_resource_info() const {
    return resource_info_;
  }

  /// Retrieve the proxy to the global cache
  proxy_type get_this_proxy();

  void pup(PUP::er& p) override;  // NOLINT

  /// Retrieve the proxy to the Main chare (or std::nullopt if the proxy has not
  /// been set, i.e. we are not charm-aware).
  std::optional<main_proxy_type> get_main_proxy();

  /// @{
  /// Wrappers for charm++ informational functions.

  /// Number of processing elements
  int number_of_procs() const;
  /// Number of nodes.
  int number_of_nodes() const;
  /// Number of processing elements on the given node.
  int procs_on_node(const int node_index) const;
  /// %Index of first processing element on the given node.
  int first_proc_on_node(const int node_index) const;
  /// %Index of the node for the given processing element.
  int node_of(const int proc_index) const;
  /// The local index for the given processing element on its node.
  int local_rank_of(const int proc_index) const;
  /// %Index of my processing element.
  int my_proc() const;
  /// %Index of my node.
  int my_node() const;
  /// The local index of my processing element on my node.
  /// This is in the interval 0, ..., procs_on_node(my_node()) - 1.
  int my_local_rank() const;
  /// @}

 private:
  // clang-tidy: false positive, redundant declaration
  template <typename GlobalCacheTag, typename MV>
  friend auto get(const GlobalCache<MV>& cache)  // NOLINT
      -> const GlobalCache_detail::type_for_get<GlobalCacheTag, MV>&;

  // clang-tidy: false positive, redundant declaration
  template <typename ParallelComponentTag, typename MV>
  friend auto get_parallel_component(  // NOLINT
      GlobalCache<MV>& cache)
      -> Parallel::proxy_from_parallel_component<
          GlobalCache_detail::get_component_if_mocked<
              typename MV::component_list, ParallelComponentTag>>&;

  // clang-tidy: false positive, redundant declaration
  template <typename ParallelComponentTag, typename MV>
  friend auto get_parallel_component(  // NOLINT
      const GlobalCache<MV>& cache)
      -> const Parallel::proxy_from_parallel_component<
          GlobalCache_detail::get_component_if_mocked<
              typename MV::component_list,
              ParallelComponentTag>>&;  // NOLINT

  ConstTagsStorage const_global_cache_{};
  MutableTagsStorage mutable_global_cache_{};
  // Wrap mutable tags in Parallel::MutexTag. The type of MutexTag is a
  // pair<mutex, mutex>. The first mutex is for editing the value of the mutable
  // tag. The second mutex is for editing the vector of callbacks associated
  // with the mutable tag.
  tuples::tagged_tuple_from_typelist<
      tmpl::transform<MutableTagsStorage, tmpl::bind<MutexTag, tmpl::_1>>>
      mutexes_{};
  ParallelComponentTuple parallel_components_{};
  Parallel::ResourceInfo<Metavariables> resource_info_{};
  bool parallel_components_have_been_set_{false};
  bool resource_info_has_been_set_{false};
  std::optional<main_proxy_type> main_proxy_;
  // Defaults for testing framework
  int my_proc_{0};
  int my_node_{0};
  int my_local_rank_{0};
  std::vector<size_t> procs_per_node_{1};
};

template <typename Metavariables>
GlobalCache<Metavariables>::GlobalCache(ConstTagsTuple const_global_cache,
                                        MutableTagsTuple mutable_global_cache,
                                        std::vector<size_t> procs_per_node,
                                        const int my_proc, const int my_node,
                                        const int my_local_rank)
    : const_global_cache_(std::move(const_global_cache)),
      mutable_global_cache_(GlobalCache_detail::make_mutable_cache_tag_storage(
          std::move(mutable_global_cache))),
      main_proxy_(std::nullopt),
      my_proc_(my_proc),
      my_node_(my_node),
      my_local_rank_(my_local_rank),
      procs_per_node_(std::move(procs_per_node)) {}

template <typename Metavariables>
GlobalCache<Metavariables>::GlobalCache(
    ConstTagsTuple const_global_cache, MutableTagsTuple mutable_global_cache,
    std::optional<main_proxy_type> main_proxy)
    : const_global_cache_(std::move(const_global_cache)),
      mutable_global_cache_(GlobalCache_detail::make_mutable_cache_tag_storage(
          std::move(mutable_global_cache))),
      main_proxy_(std::move(main_proxy)) {}

template <typename Metavariables>
void GlobalCache<Metavariables>::set_parallel_components(
    ParallelComponentTuple&& parallel_components, const CkCallback& callback) {
  ASSERT(!parallel_components_have_been_set_,
         "Can only set the parallel_components once");
  parallel_components_ = std::move(parallel_components);
  parallel_components_have_been_set_ = true;
  this->contribute(callback);
}

template <typename Metavariables>
template <typename GlobalCacheTag, typename Function>
bool GlobalCache<Metavariables>::mutable_cache_item_is_ready(
    const Parallel::ArrayComponentId& array_component_id,
    const Function& function) {
  using tag = MutableCacheTag<GlobalCache_detail::get_matching_mutable_tag<
      GlobalCacheTag, Metavariables>>;
  std::unique_ptr<Callback> optional_callback{};
  // Returns true if a callback was returned from `function`. Returns false if
  // nullptr was returned
  const auto callback_was_registered = [this, &function,
                                        &optional_callback]() -> bool {
    // Reads don't need a lock.
    if constexpr (tt::is_a_v<std::unique_ptr, typename tag::tag::type>) {
      optional_callback =
          function(*(std::get<0>(tuples::get<tag>(mutable_global_cache_))));
    } else {
      optional_callback =
          function(std::get<0>(tuples::get<tag>(mutable_global_cache_)));
    }

    return optional_callback != nullptr;
  };

  if (callback_was_registered()) {
    // Second mutex is for vector of callbacks
    std::mutex& mutex = tuples::get<MutexTag<tag>>(mutexes_).second;
    {
      // Scoped for lock guard
      const std::lock_guard<std::mutex> lock(mutex);
      std::unordered_map<Parallel::ArrayComponentId, std::unique_ptr<Callback>>&
          callbacks = std::get<1>(tuples::get<tag>(mutable_global_cache_));

      if (callbacks.count(array_component_id) != 1) {
        callbacks[array_component_id] = std::move(optional_callback);
      }
    }

    // We must check if the tag is ready again. Consider the following example:
    //
    // We have two writers, A and B preparing to make independent changes to a
    // cache object. We have an element E with a callback waiting for the change
    // B is going to make. Suppose the following sequence of events:
    //
    // 1. A mutates the object, copies the callback list (below in `mutate`),
    //    and starts the callback for element E.
    // 2. E checks the current value and determines it is not ready.
    // 3. B mutates the object, copies the empty callback list, and returns with
    //    nothing to do.
    // 4. E returns a new callback, which is added to the callback list.
    // 5. A returns.
    //
    // We now have E waiting on a change that has already happened. This will
    // most certainly result in a deadlock if E is blocking the Algorithm. Thus
    // we must do another check for whether the cache object is ready or not. In
    // the order of events, this check happens sometime after 4. When this check
    // happens, E concludes that the cache object *is* ready (because 3 is when
    // the object was mutated) and E can continue on.
    //
    // If this second check reveals that the object *is* ready, then we have an
    // unecessary callback in our map, so we remove it.
    //
    // If this second check reveals that the object *isn't* ready, then we don't
    // bother adding another callback to the map because one already exists. No
    // need to call a callback twice.
    //
    // This function returns true if no callback was registered and false if one
    // was registered.
    const bool cache_item_is_ready = not callback_was_registered();
    if (cache_item_is_ready) {
      const std::lock_guard<std::mutex> lock(mutex);
      std::unordered_map<Parallel::ArrayComponentId, std::unique_ptr<Callback>>&
          callbacks = std::get<1>(tuples::get<tag>(mutable_global_cache_));

      callbacks.erase(array_component_id);
    }

    return cache_item_is_ready;
  } else {
    // The user-defined `function` didn't specify a callback, which
    // means that the item is ready.
    return true;
  }
}

template <typename Metavariables>
template <typename GlobalCacheTag, typename Function, typename... Args>
void GlobalCache<Metavariables>::mutate(const std::tuple<Args...>& args) {
  (void)Parallel::charmxx::RegisterGlobalCacheMutate<
      Metavariables, GlobalCacheTag, Function, Args...>::registrar;
  using tag = MutableCacheTag<GlobalCache_detail::get_matching_mutable_tag<
      GlobalCacheTag, Metavariables>>;

  // Do the mutate.
  std::apply(
      [this](const auto&... local_args) {
        // First mutex is for value of mutable tag
        std::mutex& mutex = tuples::get<MutexTag<tag>>(mutexes_).first;
        const std::lock_guard<std::mutex> lock(mutex);
        if constexpr (tt::is_a_v<std::unique_ptr, typename tag::tag::type>) {
          Function::apply(make_not_null(&(*std::get<0>(
                              tuples::get<tag>(mutable_global_cache_)))),
                          local_args...);
        } else {
          Function::apply(make_not_null(&(std::get<0>(
                              tuples::get<tag>(mutable_global_cache_)))),
                          local_args...);
        }
      },
      args);

  // A callback might call mutable_cache_item_is_ready, which might add yet
  // another callback to the vector of callbacks.  We don't want to immediately
  // invoke this new callback as it might just add another callback again (and
  // again in an infinite loop). And we don't want to remove it from the map of
  // callbacks before it is invoked otherwise we could get a deadlock.
  // Therefore, after locking it, we std::move the map of callbacks into a
  // temporary map, clear the original map, and invoke the callbacks in the
  // temporary map.
  std::unordered_map<Parallel::ArrayComponentId, std::unique_ptr<Callback>>
      callbacks{};
  // Second mutex is for map of callbacks
  std::mutex& mutex = tuples::get<MutexTag<tag>>(mutexes_).second;
  {
    // Scoped for lock guard
    const std::lock_guard<std::mutex> lock(mutex);
    callbacks = std::move(std::get<1>(tuples::get<tag>(mutable_global_cache_)));
    std::get<1>(tuples::get<tag>(mutable_global_cache_)).clear();
  }

  // Invoke the callbacks.  Any new callbacks that are added to the
  // list (if a callback calls mutable_cache_item_is_ready) will be
  // saved and will not be invoked here.
  for (auto& [array_component_id, callback] : callbacks) {
    (void)array_component_id;
    callback->invoke();
  }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)
template <typename Metavariables>
void GlobalCache<Metavariables>::compute_size_for_memory_monitor(
    const double time) {
  if constexpr (tmpl::list_contains_v<
                    typename Metavariables::component_list,
                    mem_monitor::MemoryMonitor<Metavariables>>) {
    const double size_in_bytes =
        static_cast<double>(size_of_object_in_bytes(*this));
    const double size_in_MB = size_in_bytes / 1.0e6;

    auto& mem_monitor_proxy = Parallel::get_parallel_component<
        mem_monitor::MemoryMonitor<Metavariables>>(*this);

    const int my_node = Parallel::my_node<int>(*this);

    Parallel::simple_action<
        mem_monitor::ContributeMemoryData<GlobalCache<Metavariables>>>(
        mem_monitor_proxy, time, my_node, size_in_MB);
  } else {
    (void)time;
    ERROR(
        "GlobalCache::compute_size_for_memory_monitor can only be called if "
        "the MemoryMonitor is in the component list in the metavariables.\n");
  }
}

template <typename Metavariables>
void GlobalCache<Metavariables>::set_resource_info(
    const Parallel::ResourceInfo<Metavariables>& resource_info) {
  ASSERT(not resource_info_has_been_set_,
         "Can only set the resource info once");
  resource_info_ = resource_info;
  resource_info_has_been_set_ = true;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)

template <typename Metavariables>
typename Parallel::GlobalCache<Metavariables>::proxy_type
GlobalCache<Metavariables>::get_this_proxy() {
  if constexpr (is_mocked) {
    // The proxy is not used in the testing framework
    return Parallel::GlobalCache<Metavariables>::proxy_type{};
  } else {
    return this->thisProxy;
  }
}

template <typename Metavariables>
std::optional<typename Parallel::GlobalCache<Metavariables>::main_proxy_type>
GlobalCache<Metavariables>::get_main_proxy() {
  return main_proxy_;
}

// For all these functions, if the main proxy is set (meaning we are
// charm-aware) then just call the sys:: functions. Otherwise, use the values
// set for the testing framework (or the defaults).
template <typename Metavariables>
int GlobalCache<Metavariables>::number_of_procs() const {
  return main_proxy_.has_value()
             ? sys::number_of_procs()
             : static_cast<int>(alg::accumulate(procs_per_node_, 0_st));
}

template <typename Metavariables>
int GlobalCache<Metavariables>::number_of_nodes() const {
  return main_proxy_.has_value() ? sys::number_of_nodes()
                                 : static_cast<int>(procs_per_node_.size());
}

template <typename Metavariables>
int GlobalCache<Metavariables>::procs_on_node(const int node_index) const {
  return main_proxy_.has_value()
             ? sys::procs_on_node(node_index)
             : static_cast<int>(
                   procs_per_node_[static_cast<size_t>(node_index)]);
}

template <typename Metavariables>
int GlobalCache<Metavariables>::first_proc_on_node(const int node_index) const {
  return main_proxy_.has_value()
             ? sys::first_proc_on_node(node_index)
             : static_cast<int>(
                   std::accumulate(procs_per_node_.begin(),
                                   procs_per_node_.begin() + node_index, 0_st));
}

template <typename Metavariables>
int GlobalCache<Metavariables>::node_of(const int proc_index) const {
  if (main_proxy_.has_value()) {
    // For some reason gcov doesn't think this line is tested even though it is
    // in Test_AlgorithmGlobalCache.cpp
    return sys::node_of(proc_index);  // LCOV_EXCL_LINE
  } else {
    size_t procs_so_far = 0;
    size_t node = 0;
    while (procs_so_far <= static_cast<size_t>(proc_index)) {
      procs_so_far += procs_per_node_[node];
      ++node;
    }
    return static_cast<int>(--node);
  }
}

template <typename Metavariables>
int GlobalCache<Metavariables>::local_rank_of(const int proc_index) const {
  return main_proxy_.has_value()
             ? sys::local_rank_of(proc_index)
             : proc_index - first_proc_on_node(node_of(proc_index));
}

template <typename Metavariables>
int GlobalCache<Metavariables>::my_proc() const {
  return main_proxy_.has_value() ? sys::my_proc() : my_proc_;
}

template <typename Metavariables>
int GlobalCache<Metavariables>::my_node() const {
  return main_proxy_.has_value() ? sys::my_node() : my_node_;
}

template <typename Metavariables>
int GlobalCache<Metavariables>::my_local_rank() const {
  return main_proxy_.has_value() ? sys::my_local_rank() : my_local_rank_;
}

template <typename Metavariables>
void GlobalCache<Metavariables>::pup(PUP::er& p) {
  p | const_global_cache_;
  p | parallel_components_;
  p | mutable_global_cache_;
  p | main_proxy_;
  p | parallel_components_have_been_set_;
  p | resource_info_has_been_set_;
  p | my_proc_;
  p | my_node_;
  p | my_local_rank_;
  p | procs_per_node_;
}

/// @{
/// \ingroup ParallelGroup
/// \brief Access the Charm++ proxy associated with a ParallelComponent
///
/// \requires ParallelComponentTag is a tag in component_list
///
/// \returns a Charm++ proxy that can be used to call an entry method on the
/// chare(s)
template <typename ParallelComponentTag, typename Metavariables>
auto get_parallel_component(GlobalCache<Metavariables>& cache)
    -> Parallel::proxy_from_parallel_component<
        GlobalCache_detail::get_component_if_mocked<
            typename Metavariables::component_list, ParallelComponentTag>>& {
  return tuples::get<tmpl::type_<Parallel::proxy_from_parallel_component<
      GlobalCache_detail::get_component_if_mocked<
          typename Metavariables::component_list, ParallelComponentTag>>>>(
      cache.parallel_components_);
}

template <typename ParallelComponentTag, typename Metavariables>
auto get_parallel_component(const GlobalCache<Metavariables>& cache)
    -> const Parallel::proxy_from_parallel_component<
        GlobalCache_detail::get_component_if_mocked<
            typename Metavariables::component_list, ParallelComponentTag>>& {
  return tuples::get<tmpl::type_<Parallel::proxy_from_parallel_component<
      GlobalCache_detail::get_component_if_mocked<
          typename Metavariables::component_list, ParallelComponentTag>>>>(
      cache.parallel_components_);
}
/// @}

/// @{
/// \ingroup ParallelGroup
/// \brief Access data in the cache
///
/// \requires GlobalCacheTag is a tag in the `mutable_global_cache_tags`
/// or `const_global_cache_tags` defined by the Metavariables and in Actions.
///
/// \returns a constant reference to an object in the cache
template <typename GlobalCacheTag, typename Metavariables>
auto get(const GlobalCache<Metavariables>& cache)
    -> const GlobalCache_detail::type_for_get<GlobalCacheTag, Metavariables>& {
  constexpr bool is_mutable =
      is_in_mutable_global_cache<Metavariables, GlobalCacheTag>;
  // We check if the tag is to be retrieved directly or via a base class
  using tmp_tag =
      GlobalCache_detail::get_matching_tag<GlobalCacheTag, Metavariables>;
  using tag =
      tmpl::conditional_t<is_mutable, MutableCacheTag<tmp_tag>, tmp_tag>;
  if constexpr (is_mutable) {
    // Tag is not in the const tags, so use mutable_global_cache_. No locks here
    // because we require all mutable tags to be able to be read at all times
    // (even when being written to)
    if constexpr (tt::is_a_v<std::unique_ptr, typename tag::tag::type>) {
      return *std::get<0>(tuples::get<tag>(cache.mutable_global_cache_));
    } else {
      return std::get<0>(tuples::get<tag>(cache.mutable_global_cache_));
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
/// `std::unique_ptr<CallBack>` that determines the readiness. To
/// indicate that the item is ready, the `std::unique_ptr` returned
/// by `function` must be nullptr; in this case
/// `mutable_cache_item_is_ready` returns true. To indicate that the
/// item is not ready, the `std::unique_ptr` returned by `function`
/// must be valid; in this case, `mutable_cache_item_is_ready`
/// appends the `std::unique_ptr<Callback>` to an
/// internal list of callbacks to be called on `mutate`, and then
/// returns false.
///
/// \note If `function` is returning a valid callback, it should only return a
/// `Parallel::PerformAlgorithmCallback`. Other types of callbacks are not
/// supported at this time.
template <typename GlobalCacheTag, typename Function, typename Metavariables>
bool mutable_cache_item_is_ready(
    GlobalCache<Metavariables>& cache,
    const Parallel::ArrayComponentId& array_component_id,
    const Function& function) {
  return cache.template mutable_cache_item_is_ready<GlobalCacheTag>(
      array_component_id, function);
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
template <typename GlobalCacheTag, typename Function, typename Metavariables,
          typename... Args>
void mutate(GlobalCache<Metavariables>& cache, Args&&... args) {
  if (cache.get_main_proxy().has_value()) {
    if constexpr (not GlobalCache<Metavariables>::is_mocked) {
      cache.thisProxy.template mutate<GlobalCacheTag, Function>(
          std::make_tuple<Args...>(std::forward<Args>(args)...));
    } else {
      ERROR(
          "Main proxy is set but global cache is being mocked. This is "
          "currently not implemented.");
    }
  } else {
    cache.template mutate<GlobalCacheTag, Function>(
        std::make_tuple<Args...>(std::forward<Args>(args)...));
  }
}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag to retrieve the `Parallel::GlobalCache` from the DataBox.
struct GlobalCache : db::BaseTag {};

template <class Metavariables>
struct GlobalCacheProxy : db::SimpleTag {
  using type = CProxy_GlobalCache<Metavariables>;
};

template <class Metavariables>
struct GlobalCacheImpl : GlobalCache, db::SimpleTag {
  using type = Parallel::GlobalCache<Metavariables>*;
  static std::string name() { return "GlobalCache"; }
};

template <class Metavariables>
struct GlobalCacheImplCompute : GlobalCacheImpl<Metavariables>, db::ComputeTag {
  using base = GlobalCacheImpl<Metavariables>;
  using argument_tags = tmpl::list<GlobalCacheProxy<Metavariables>>;
  using return_type = Parallel::GlobalCache<Metavariables>*;
  static void function(
      const gsl::not_null<Parallel::GlobalCache<Metavariables>**>
          local_branch_of_global_cache,
      const CProxy_GlobalCache<Metavariables>& global_cache_proxy) {
    *local_branch_of_global_cache = Parallel::local_branch(global_cache_proxy);
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag used to retrieve data from the `Parallel::GlobalCache`. This is the
/// recommended way for compute tags to retrieve data out of the global cache.
template <class CacheTag>
struct FromGlobalCache : CacheTag, db::ReferenceTag {
  static_assert(db::is_simple_tag_v<CacheTag>);
  using base = CacheTag;
  using argument_tags = tmpl::list<GlobalCache>;

  template <class Metavariables>
  static const auto& get(
      const Parallel::GlobalCache<Metavariables>* const& cache) {
    return Parallel::get<CacheTag>(*cache);
  }
};

template <typename Metavariables>
struct ResourceInfoReference : ResourceInfo<Metavariables>, db::ReferenceTag {
  using base = ResourceInfo<Metavariables>;
  using argument_tags = tmpl::list<GlobalCache>;

  static const auto& get(
      const Parallel::GlobalCache<Metavariables>* const& cache) {
    return cache->get_resource_info();
  }
};
}  // namespace Tags
}  // namespace Parallel

namespace PUP {
/// \cond
// Warning! This is an invalid and kludgey pupper, because when unpacking it
// produces a `nullptr` rather than a valid `GlobalCache*`. This function is
// provided _only_ to enable putting a `GlobalCache*` in the DataBox.
//
// SpECTRE parallel components with a `GlobalCache*` in their DataBox should
// set this pointer using a compute item that calls `Parallel::local_branch` on
// a stored proxy. When deserializing the DataBox, these components should call
// `db:mutate<GlobalCacheProxy>(box)` to force the DataBox to update the
// pointer from the new Charm++ proxy.
//
// We do not currently anticipate needing to (de)serialize a `GlobalCache*`
// outside of a DataBox. But if this need arises, it will be necessary to
// provide a non-kludgey pupper here.
//
// Correctly (de)serializing the `GlobalCache*` would require obtaining a
// `CProxy_GlobalCache` item and calling `Parallel::local_branch` on it ---
// just as in the workflow described above, but within the pupper vs in the
// DataBox. But this strategy fails when restarting from a checkpoint file,
// because calling `Parallel::local_branch` may not be well-defined in the
// unpacking pup call when all Charm++ components may not yet be fully restored.
// This difficulty is why we instead write an invalid pupper here.
//
// In future versions of Charm++, the pup function may know whether it is
// called in the context of checkpointing, load balancing, etc. This knowledge
// would enable us to write a valid pupper for non-checkpoint contexts, and
// return a `nullptr` only when restoring from checkpoint.
template <typename Metavariables>
inline void operator|(PUP::er& p,  // NOLINT
                      Parallel::GlobalCache<Metavariables>*& t) {
  if (p.isUnpacking()) {
    t = nullptr;
  }
}
/// \endcond
}  // namespace PUP

#define CK_TEMPLATES_ONLY
#include "Parallel/GlobalCache.def.h"
#undef CK_TEMPLATES_ONLY
