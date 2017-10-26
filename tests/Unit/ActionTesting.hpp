// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ActionTesting_detail {
template <typename InboxTagList>
class MockArrayElementProxy {
 public:
  using Inbox = tuples::TaggedTupleTypelist<InboxTagList>;

  explicit MockArrayElementProxy(Inbox& inbox) : inbox_(inbox) {}

  template <typename InboxTag, typename Data>
  void receive_data(const typename InboxTag::temporal_id& id,
                    const Data& data) {
    tuples::get<InboxTag>(inbox_)[id].emplace(data);
  }

 private:
  Inbox& inbox_;
};

template <typename Index, typename InboxTagList>
class MockProxy {
 public:
  using Inboxes =
      std::unordered_map<Index, tuples::TaggedTupleTypelist<InboxTagList>>;

  MockProxy() : inboxes_(nullptr) {}
  void set_inboxes(Inboxes* inboxes) { inboxes_ = inboxes; }

  MockArrayElementProxy<InboxTagList> operator[](const Index& index) {
    return {(*inboxes_)[index]};
  }

 private:
  Inboxes* inboxes_;
};

struct MockArrayChare {
  template <typename Component, typename Metavariables, typename ActionList,
            typename InboxTagList, typename Index, typename InitialDataBox>
  using cproxy = MockProxy<Index, InboxTagList>;
};
}  // namespace ActionTesting_detail

/// \cond HIDDEN_SYMBOLS
namespace Parallel {
template <>
struct get_array_index<ActionTesting_detail::MockArrayChare> {
  template <typename Component>
  using f = typename Component::index;
};
}  // namespace Parallel
/// \endcond

namespace ActionTesting {
/// \ingroup TestingFramework
/// A mock parallel component that acts like a component with
/// chare_type Parallel::Algorithms::Array.
template <typename Metavariables, typename Index,
          typename ConstGlobalCacheTagList,
          typename InboxTagList = tmpl::list<>,
          typename ActionList = tmpl::list<>>
struct MockArrayComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting_detail::MockArrayChare;
  using index = Index;
  using const_global_cache_tag_list = ConstGlobalCacheTagList;
  using inbox_tag_list = InboxTagList;
  using action_list = ActionList;
  using initial_databox = NoSuchType;
};

/// \ingroup TestingFramework
/// A class that mocks the infrastructure needed to run actions.  It
/// simulates message passing using the inbox infrastructure and
/// handles most of the arguments to the apply and is_ready action
/// methods.
template <typename Metavariables>
class ActionRunner {
 public:
  // No moving, since MockProxy holds a pointer to us.
  ActionRunner(const ActionRunner&) = delete;
  ActionRunner(ActionRunner&&) = delete;
  ActionRunner& operator=(const ActionRunner&) = delete;
  ActionRunner& operator=(ActionRunner&&) = delete;
  ~ActionRunner() = default;

  using ConstGlobalCache_t = Parallel::ConstGlobalCache<Metavariables>;
  using CacheTuple_t =
      tuples::TaggedTupleTypelist<typename ConstGlobalCache_t::tag_list>;

  /// Construct from the tuple of ConstGlobalCache objects.
  explicit ActionRunner(CacheTuple_t cache_contents)
      : cache_(std::move(cache_contents)) {
    tmpl::for_each<typename Metavariables::component_list>(
        [this](auto component) {
          using Component = tmpl::type_from<decltype(component)>;
          Parallel::get_parallel_component<Component>(cache_)
              .set_inboxes(&tuples::get<Inboxes<Component>>(inboxes_));
        });
  }

  /// Call Action::apply as if on the portion of Component labeled by
  /// array_index.
  template <typename Component, typename Action, typename DbTags>
  decltype(auto) apply(db::DataBox<DbTags>& box,
                       const typename Component::index& array_index) noexcept {
    return Action::apply(box, inboxes<Component>()[array_index], cache_,
                         array_index, typename Component::action_list{});
  }

  /// Call Action::is_ready as if on the portion of Component labeled
  /// by array_index.
  template <typename Component, typename Action, typename DbTags>
  bool is_ready(const db::DataBox<DbTags>& box,
                const typename Component::index& array_index) noexcept {
    return Action::is_ready(box, as_const(inboxes<Component>()[array_index]),
                            cache_, array_index);
  }

  /// Access the inboxes for a given component.
  template <typename Component>
  std::unordered_map<
      typename Component::index,
      tuples::TaggedTupleTypelist<typename Component::inbox_tag_list>>&
  inboxes() noexcept {
    return tuples::get<Inboxes<Component>>(inboxes_);
  }

  /// Find the set of array indices on Component where the specified
  /// inbox is not empty.
  template <typename Component, typename InboxTag>
  std::unordered_set<typename Component::index>
  nonempty_inboxes() noexcept {
    std::unordered_set<typename Component::index> result;
    for (const auto& element_box : inboxes<Component>()) {
      if (not tuples::get<InboxTag>(element_box.second).empty()) {
        result.insert(element_box.first);
      }
    }
    return result;
  }

 private:
  template <typename Component>
  struct Inboxes {
    using type = std::unordered_map<
      typename Component::index,
      tuples::TaggedTupleTypelist<typename Component::inbox_tag_list>>;
  };

  ConstGlobalCache_t cache_;
  tuples::TaggedTupleTypelist<
    tmpl::transform<typename Metavariables::component_list,
                    tmpl::bind<Inboxes, tmpl::_1>>>
      inboxes_;
};
}  // namespace ActionTesting
