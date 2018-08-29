// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/SimpleActionVisitation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ActionTesting {
// MockLocalAlgorithm mocks the AlgorithmImpl class.
template <typename Component>
class MockLocalAlgorithm {
 public:
  using actions_list = typename Component::action_list;

  using inbox_tags_list = Parallel::get_inbox_tags<actions_list>;

  using metavariables = typename Component::metavariables;

 private:
  template <typename ActiontList>
  struct compute_databox_type;

  template <typename... ActionsPack>
  struct compute_databox_type<tmpl::list<ActionsPack...>> {
    using type = Parallel::Algorithm_detail::build_action_return_typelist<
        typename Component::initial_databox,
        tmpl::list<
            tuples::TaggedTupleTypelist<inbox_tags_list>,
            Parallel::ConstGlobalCache<typename Component::metavariables>,
            typename Component::array_index, actions_list,
            std::add_pointer_t<Component>>,
        ActionsPack...>;
  };

 public:
  using databox_types = typename compute_databox_type<actions_list>::type;

  MockLocalAlgorithm() = default;

  explicit MockLocalAlgorithm(typename Component::initial_databox initial_box)
      : box_(std::move(initial_box)) {}

  void set_index(typename Component::index index) noexcept {
    array_index_ = std::move(index);
  }

  void set_cache(Parallel::ConstGlobalCache<typename Component::metavariables>*
                     cache_ptr) noexcept {
    const_global_cache_ = cache_ptr;
  }

  void set_inboxes(tuples::TaggedTupleTypelist<
                   Parallel::get_inbox_tags<typename Component::action_list>>*
                       inboxes_ptr) noexcept {
    inboxes_ = inboxes_ptr;
  }

  void set_terminate(bool t) { terminate_ = t; }
  bool get_terminate() { return terminate_; }

  template <typename BoxType>
  BoxType& get_databox() noexcept {
    return boost::get<BoxType>(box_);
  }

  template <typename BoxType>
  const BoxType& get_databox() const noexcept {
    return boost::get<BoxType>(box_);
  }

  auto& get_variant_box() noexcept { return box_; }

  const auto& get_variant_box() const noexcept { return box_; }

  void force_next_action_to_be(const size_t next_action_id) noexcept {
    algorithm_step_ = next_action_id;
  }

  template <size_t... Is>
  void next_action(std::index_sequence<Is...> /*meta*/) noexcept;

  template <size_t... Is>
  bool is_ready(
      std::index_sequence<Is...> /*meta*/) noexcept;

  template <typename Action, typename... Args>
  void simple_action(std::tuple<Args...> args) noexcept {
    performing_action_ = true;
    forward_tuple_to_action<Action>(
        std::move(args), std::make_index_sequence<sizeof...(Args)>{});
    performing_action_ = false;
  }

  template <typename Action>
  void simple_action() noexcept {
    performing_action_ = true;
    Parallel::Algorithm_detail::simple_action_visitor<
        Action, typename Component::initial_databox>(
        box_, *inboxes_, *const_global_cache_, cpp17::as_const(array_index_),
        actions_list{}, std::add_pointer_t<Component>{nullptr});
    performing_action_ = false;
  }

 private:
  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_action(std::tuple<Args...>&& args,
                               std::index_sequence<Is...> /*meta*/) noexcept {
    Parallel::Algorithm_detail::simple_action_visitor<
        Action, typename Component::initial_databox>(
        box_, *inboxes_, *const_global_cache_, cpp17::as_const(array_index_),
        actions_list{}, std::add_pointer_t<Component>{nullptr},
        std::forward<Args>(std::get<Is>(args))...);
  }

  bool terminate_{false};
  make_boost_variant_over<
      tmpl::push_front<databox_types, db::DataBox<tmpl::list<>>>>
      box_ = db::DataBox<tmpl::list<>>{};
  // The next action we should execute.
  size_t algorithm_step_ = 0;
  bool performing_action_ = false;

  typename Component::index array_index_{};
  Parallel::ConstGlobalCache<typename Component::metavariables>*
      const_global_cache_{nullptr};
  tuples::TaggedTupleTypelist<
      Parallel::get_inbox_tags<typename Component::action_list>>* inboxes_{
      nullptr};
};

template <typename Component>
template <size_t... Is>
void MockLocalAlgorithm<Component>::next_action(
    std::index_sequence<Is...> /*meta*/) noexcept {
  auto& const_global_cache = *const_global_cache_;
  if (performing_action_) {
    ERROR(
        "Cannot call an Action while already calling an Action on the same "
        "MockLocalAlgorithm (an element of a parallel component array, or a "
        "parallel component singleton).");
  }
  // Keep track of if we already evaluated an action since we want `next_action`
  // to only evaluate one per call.
  bool already_did_an_action = false;
  const auto helper = [
    this, &array_index = array_index_, &inboxes = *inboxes_,
    &const_global_cache, &already_did_an_action
  ](auto iteration) noexcept {
    constexpr size_t iter = decltype(iteration)::value;
    using this_action = tmpl::at_c<actions_list, iter>;
    using this_databox =
        tmpl::at_c<databox_types,
                   iter == 0 ? tmpl::size<databox_types>::value - 1 : iter>;
    if (already_did_an_action or algorithm_step_ != iter) {
      return;
    }

    this_databox* box_ptr{};
    try {
      box_ptr = &boost::get<this_databox>(box_);
    } catch (std::exception& e) {
      ERROR(
          "\nFailed to retrieve Databox in take_next_action:\nCaught "
          "exception: '"
          << e.what() << "'\nDataBox type: '"
          << pretty_type::get_name<this_databox>() << "'\nIteration: " << iter
          << "\nAction: '" << pretty_type::get_name<this_action>()
          << "'\nBoost::Variant id: " << box_.which()
          << "\nBoost::Variant type is: '" << type_of_current_state(box_)
          << "'\n\n");
    }
    this_databox& box = *box_ptr;

    const auto check_if_ready = make_overloader(
        [&box, &array_index, &const_global_cache, &inboxes](
            std::true_type /*has_is_ready*/, auto t) {
          return decltype(t)::is_ready(
              cpp17::as_const(box), cpp17::as_const(inboxes),
              const_global_cache, cpp17::as_const(array_index));
        },
        [](std::false_type /*has_is_ready*/, auto) { return true; });
    if (not check_if_ready(Parallel::Algorithm_detail::is_is_ready_callable_t<
                               this_action, this_databox,
                               tuples::TaggedTupleTypelist<inbox_tags_list>,
                               Parallel::ConstGlobalCache<metavariables>,
                               typename Component::index>{},
                           this_action{})) {
      ERROR("Tried to invoke the action '"
            << pretty_type::get_name<this_action>()
            << "' but have not received all the "
               "necessary data.");
    }
    performing_action_ = true;
    algorithm_step_++;
    constexpr Component const* const component_ptr = nullptr;
    make_overloader(
        [ this, &array_index, component_ptr, &const_global_cache, &inboxes ](
            auto& my_box, std::integral_constant<size_t, 1> /*meta*/)
            SPECTRE_JUST_ALWAYS_INLINE noexcept {
              std::tie(box_) = this_action::apply(
                  my_box, inboxes, const_global_cache,
                  cpp17::as_const(array_index), actions_list{}, component_ptr);
            },
        [ this, &array_index, component_ptr, &const_global_cache, &inboxes ](
            auto& my_box, std::integral_constant<size_t, 2> /*meta*/)
            SPECTRE_JUST_ALWAYS_INLINE noexcept {
              std::tie(box_, terminate_) = this_action::apply(
                  my_box, inboxes, const_global_cache,
                  cpp17::as_const(array_index), actions_list{}, component_ptr);
            },
        [ this, &array_index, component_ptr, &const_global_cache, &inboxes ](
            auto& my_box, std::integral_constant<size_t, 3> /*meta*/)
            SPECTRE_JUST_ALWAYS_INLINE noexcept {
              std::tie(box_, terminate_, algorithm_step_) = this_action::apply(
                  my_box, inboxes, const_global_cache,
                  cpp17::as_const(array_index), actions_list{}, component_ptr);
            })(
        box,
        typename std::tuple_size<decltype(this_action::apply(
            box, inboxes, const_global_cache, cpp17::as_const(array_index),
            actions_list{}, component_ptr))>::type{});
    performing_action_ = false;
    already_did_an_action = true;

    // Wrap counter if necessary
    if (algorithm_step_ >= tmpl::size<actions_list>::value) {
      algorithm_step_ = 0;
    }
  };
  // Silence compiler warning when there are no Actions.
  (void)helper;
  EXPAND_PACK_LEFT_TO_RIGHT(helper(std::integral_constant<size_t, Is>{}));
}

template <typename Component>
template <size_t... Is>
bool MockLocalAlgorithm<Component>::is_ready(
    std::index_sequence<Is...> /*meta*/) noexcept {
  bool next_action_is_ready = false;
  const auto helper = [
    this, &array_index = array_index_, &inboxes = *inboxes_,
    &const_global_cache = const_global_cache_, &next_action_is_ready
  ](auto iteration) noexcept {
    constexpr size_t iter = decltype(iteration)::value;
    using this_action = tmpl::at_c<actions_list, iter>;
    using this_databox =
        tmpl::at_c<databox_types,
                   iter == 0 ? tmpl::size<databox_types>::value - 1 : iter>;
    if (iter != algorithm_step_) {
      return;
    }

    this_databox* box_ptr{};
    try {
      box_ptr = &boost::get<this_databox>(box_);
    } catch (std::exception& e) {
      ERROR(
          "\nFailed to retrieve Databox in take_next_action:\nCaught "
          "exception: '"
          << e.what() << "'\nDataBox type: '"
          << pretty_type::get_name<this_databox>() << "'\nIteration: " << iter
          << "\nAction: '" << pretty_type::get_name<this_action>()
          << "'\nBoost::Variant id: " << box_.which()
          << "\nBoost::Variant type is: '" << type_of_current_state(box_)
          << "'\n\n");
    }
    this_databox& box = *box_ptr;

    const auto check_if_ready = make_overloader(
        [&box, &array_index, &const_global_cache, &inboxes](
            std::true_type /*has_is_ready*/, auto t) {
          return decltype(t)::is_ready(
              cpp17::as_const(box), cpp17::as_const(inboxes),
              *const_global_cache, cpp17::as_const(array_index));
        },
        [](std::false_type /*has_is_ready*/, auto) { return true; });

    next_action_is_ready =
        check_if_ready(Parallel::Algorithm_detail::is_is_ready_callable_t<
                           this_action, this_databox,
                           tuples::TaggedTupleTypelist<inbox_tags_list>,
                           Parallel::ConstGlobalCache<metavariables>,
                           typename Component::index>{},
                       this_action{});
  };
  // Silence compiler warning when there are no Actions.
  (void)helper;
  EXPAND_PACK_LEFT_TO_RIGHT(helper(std::integral_constant<size_t, Is>{}));
  return next_action_is_ready;
}

namespace ActionTesting_detail {
template <typename Component, typename InboxTagList>
class MockArrayElementProxy {
 public:
  using Inbox = tuples::TaggedTupleTypelist<InboxTagList>;

  MockArrayElementProxy(MockLocalAlgorithm<Component>& local_algorithm,
                        Inbox& inbox)
      : local_algorithm_(local_algorithm), inbox_(inbox) {}

  template <typename InboxTag, typename Data>
  void receive_data(const typename InboxTag::temporal_id& id, const Data& data,
                    const bool enable_if_disabled = false) {
    // Might be useful in the future, not needed now but required by the
    // interface to be compliant with the Algorithm invocations.
    (void)enable_if_disabled;
    tuples::get<InboxTag>(inbox_)[id].emplace(data);
  }

  MockLocalAlgorithm<Component>* ckLocal() { return &local_algorithm_; }

 private:
  MockLocalAlgorithm<Component>& local_algorithm_;
  Inbox& inbox_;
};

template <typename Component, typename Index, typename InboxTagList>
class MockProxy {
 public:
  using Inboxes =
      std::unordered_map<Index, tuples::TaggedTupleTypelist<InboxTagList>>;
  using LocalAlgorithms =
      std::unordered_map<Index, MockLocalAlgorithm<Component>>;

  MockProxy() : inboxes_(nullptr) {}

  void set_data(LocalAlgorithms* local_algorithms,
                Inboxes* inboxes) {
    local_algorithms_ = local_algorithms;
    inboxes_ = inboxes;
  }

  MockArrayElementProxy<Component, InboxTagList> operator[](
      const Index& index) {
    ASSERT(local_algorithms_->count(index) == 1,
           "Should have exactly one local algorithm with key '"
               << index << "' but found " << local_algorithms_->count(index)
               << ". The known keys are " << keys_of(*local_algorithms_)
               << ". Did you forget to add a local algorithm when constructing "
                  "the ActionRunner?");
    return MockArrayElementProxy<Component, InboxTagList>(
        local_algorithms_->at(index), inboxes_->operator[](index));
  }

  MockLocalAlgorithm<Component>* ckLocalBranch() noexcept {
    ASSERT(
        local_algorithms_->size() == 1,
        "Can only have one algorithm when getting the ckLocalBranch, but have "
            << local_algorithms_->size());
    // We always retrieve the 0th local branch because we are assuming running
    // on a single core.
    return std::addressof(local_algorithms_->at(0));
  }

  // clang-tidy: no non-const references
  void pup(PUP::er& /*p*/) noexcept {  // NOLINT
    ERROR(
        "Should not try to serialize the mock proxy. If you encountered this "
        "error you are using the mocking framework in a way that it was not "
        "intended to be used. It may be possible to extend it to more use "
        "cases but it is recommended you file an issue to discuss before "
        "modifying the mocking framework.");
  }

 private:
  LocalAlgorithms* local_algorithms_;
  Inboxes* inboxes_;
};

struct MockArrayChare {
  template <typename Component, typename Metavariables, typename ActionList,
            typename Index, typename InitialDataBox>
  using cproxy =
      MockProxy<Component, Index, Parallel::get_inbox_tags<ActionList>>;
};
}  // namespace ActionTesting_detail
}  // namespace ActionTesting

/// \cond HIDDEN_SYMBOLS
namespace Parallel {
template <>
struct get_array_index<ActionTesting::ActionTesting_detail::MockArrayChare> {
  template <typename Component>
  using f = typename Component::index;
};
}  // namespace Parallel
/// \endcond

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Structures used for mocking the parallel components framework in order
 * to test actions.
 */
namespace ActionTesting {
/// \ingroup TestingFrameworkGroup
/// A mock parallel component that acts like a component with
/// chare_type Parallel::Algorithms::Array.
template <typename Metavariables, typename Index,
          typename ConstGlobalCacheTagList, typename ActionList = tmpl::list<>,
          typename ComponentBeingMocked = void>
struct MockArrayComponent {
  // We need a way of grabbing the correct proxy from the ConstGlobalCache. The
  // way we do that is by checking if the component passed to
  // `Parallel::get_parallel_component` is in the
  // `Metavariables::component_list`, if not then we search for the element of
  // `Metavariables::component_list` that has `component_being_mocked` equal to
  // the `ParallelComponentTag` passed to `Parallel::get_parallel_component`.
  using component_being_mocked = ComponentBeingMocked;

  using metavariables = Metavariables;
  using chare_type = ActionTesting_detail::MockArrayChare;
  using index = Index;
  using array_index = Index;
  using const_global_cache_tag_list = ConstGlobalCacheTagList;
  using action_list = ActionList;
};

/// \ingroup TestingFrameworkGroup
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

  template <typename Component>
  struct InboxesTag {
    using type =
        std::unordered_map<typename Component::index,
                           tuples::TaggedTupleTypelist<Parallel::get_inbox_tags<
                               typename Component::action_list>>>;
  };

  template <typename Component>
  struct LocalAlgorithmsTag {
    using type = std::unordered_map<typename Component::index,
                                    MockLocalAlgorithm<Component>>;
  };

  using GlobalCache = Parallel::ConstGlobalCache<Metavariables>;
  using CacheTuple =
      tuples::TaggedTupleTypelist<typename GlobalCache::tag_list>;
  using LocalAlgorithms = tuples::TaggedTupleTypelist<
      tmpl::transform<typename Metavariables::component_list,
                      tmpl::bind<LocalAlgorithmsTag, tmpl::_1>>>;
  using Inboxes = tuples::TaggedTupleTypelist<
      tmpl::transform<typename Metavariables::component_list,
                      tmpl::bind<InboxesTag, tmpl::_1>>>;

  /// Construct from the tuple of ConstGlobalCache objects.
  explicit ActionRunner(CacheTuple cache_contents,
                        LocalAlgorithms local_algorithms)
      : cache_(std::move(cache_contents)),
        local_algorithms_(std::move(local_algorithms)) {
    tmpl::for_each<typename Metavariables::component_list>(
        [this](auto component) {
          using Component = tmpl::type_from<decltype(component)>;
          Parallel::get_parallel_component<Component>(cache_).set_data(
              &tuples::get<LocalAlgorithmsTag<Component>>(local_algorithms_),
              &tuples::get<InboxesTag<Component>>(inboxes_));

          for (auto& local_alg_pair : this->template algorithms<Component>()) {
            const auto& index = local_alg_pair.first;
            auto& local_alg = local_alg_pair.second;
            local_alg.set_index(index);
            local_alg.set_cache(&cache_);
            local_alg.set_inboxes(
                &(tuples::get<InboxesTag<Component>>(inboxes_)[index]));
          }
        });
  }

  /// Invoke the simple action `Action` on the `Component` labeled by
  /// `array_index`
  template <typename Component, typename Action, typename... Args>
  void simple_action(const typename Component::index& array_index,
                     Args&&... args) noexcept {
    Parallel::Algorithm_detail::simple_action_visitor<
        Action, typename Component::initial_databox>(
        algorithms<Component>().at(array_index).get_variant_box(),
        inboxes<Component>()[array_index], cache_, array_index,
        typename Component::action_list{},
        std::add_pointer_t<Component>{nullptr}, std::forward<Args>(args)...);
  }

  /// Instead of the next call to `next_action` applying the next action in
  /// the action list, force the next action to be `Action`
  template <typename Component, typename Action>
  void force_next_action_to_be(
      const typename Component::index& array_index) noexcept {
    static_assert(
        tmpl::list_contains_v<typename Component::action_list, Action>,
        "Cannot force a next action that is not in the action list of the "
        "parallel component. See the first template parameter of "
        "'force_next_action_to_be' for the component and the second template "
        "parameter for the action.");
    algorithms<Component>()
        .at(array_index)
        .force_next_action_to_be(
            tmpl::index_of<typename Component::action_list, Action>::value);
  }

  /// Invoke the next action in the ActionList on the parallel component
  /// `Component` on the component labeled by `array_index`.
  template <typename Component>
  void next_action(const typename Component::index& array_index) noexcept {
    algorithms<Component>()
        .at(array_index)
        .next_action(
            std::make_index_sequence<tmpl::size<typename MockLocalAlgorithm<
                Component>::actions_list>::value>{});
  }

  /// Call is_ready on the next action in the action list as if on the portion
  /// of Component labeled by array_index.
  template <typename Component>
  bool is_ready(const typename Component::index& array_index) noexcept {
    return algorithms<Component>()
        .at(array_index)
        .is_ready(
            std::make_index_sequence<tmpl::size<typename MockLocalAlgorithm<
                Component>::actions_list>::value>{});
  }

  /// Access the inboxes for a given component.
  template <typename Component>
  std::unordered_map<typename Component::index,
                     tuples::TaggedTupleTypelist<Parallel::get_inbox_tags<
                         typename Component::action_list>>>&
  inboxes() noexcept {
    return tuples::get<InboxesTag<Component>>(inboxes_);
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

  /// Access the mocked algorithms for a component, indexed by array index.
  template <typename Component>
  auto& algorithms() noexcept {
    return tuples::get<LocalAlgorithmsTag<Component>>(local_algorithms_);
  }

  const GlobalCache& cache() noexcept { return cache_; }

 private:
  GlobalCache cache_;
  Inboxes inboxes_;
  LocalAlgorithms local_algorithms_;
};
}  // namespace ActionTesting
