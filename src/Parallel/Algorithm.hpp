// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <functional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/Error.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/TypeTraits.hpp"
#include "Utilities/BoostHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
namespace charmxx {
/*!
 * Uses the __PRETTY_FUNCTION__ compiler intrinsic to extract the template
 * parameter names in the same form that Charm++ uses to register entry methods.
 * This is used by the generated Singleton, Array, Group and Nodegroup headers,
 * as well as in CharmMain.cpp.
 */
template <class... Args>
std::string get_template_parameters_as_string() {
  std::string function_name(static_cast<char const*>(__PRETTY_FUNCTION__));
  std::string template_params =
      function_name.substr(function_name.find(std::string("Args = ")) + 8);
  template_params.erase(template_params.end() - 2, template_params.end());
  size_t pos = 0;
  while ((pos = template_params.find(" >")) != std::string::npos) {
    template_params.replace(pos, 1, ">");
    template_params.erase(pos + 1, 1);
  }
  pos = 0;
  while ((pos = template_params.find(", ", pos)) != std::string::npos) {
    template_params.erase(pos + 1, 1);
  }
  pos = 0;
  while ((pos = template_params.find('>', pos + 2)) != std::string::npos) {
    template_params.replace(pos, 1, " >");
  }
  std::replace(template_params.begin(), template_params.end(), '%', '>');
  // GCC's __PRETTY_FUNCTION__ adds the return type at the end, so we remove it.
  if (template_params.find('}') != std::string::npos) {
    template_params.erase(template_params.find('}'), template_params.size());
  }
  return template_params;
}
}  // namespace charmxx

/*!
 * \ingroup ParallelGroup
 * \brief Lock a converse CmiNodeLock
 */
inline void lock(const gsl::not_null<CmiNodeLock*> node_lock) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CmiLock(*node_lock);
#pragma GCC diagnostic pop
}

/// \cond
constexpr inline void lock(
    const gsl::not_null<NoSuchType*> /*unused*/) noexcept {}
/// \endcond

/*!
 * \ingroup ParallelGroup
 * \brief Returns true if the lock was successfully acquired and false if the
 * lock is already acquired by another processor.
 */
inline bool try_lock(const gsl::not_null<CmiNodeLock*> node_lock) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  return CmiTryLock(*node_lock) == 0;
#pragma GCC diagnostic pop
}

/*!
 * \ingroup ParallelGroup
 * \brief Unlock a converse CmiNodeLock
 */
inline void unlock(const gsl::not_null<CmiNodeLock*> node_lock) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CmiUnlock(*node_lock);
#pragma GCC diagnostic pop
}

/// \cond
constexpr inline void unlock(
    const gsl::not_null<NoSuchType*> /*unused*/) noexcept {}
/// \\endcond

namespace Algorithm_detail {
template <bool, typename AdditionalArgsList>
struct build_action_return_types_impl;

template <typename... AdditionalArgs>
struct build_action_return_types_impl<false, tmpl::list<AdditionalArgs...>> {
  template <typename LastReturnType, typename ReturnTypeList>
  using f = tmpl::push_back<ReturnTypeList, LastReturnType>;
};

template <typename... AdditionalArgs>
struct build_action_return_types_impl<true, tmpl::list<AdditionalArgs...>> {
  template <typename LastReturnType, typename ReturnTypeList, typename Action,
            typename... Actions>
  using f = typename build_action_return_types_impl<
      sizeof...(Actions) != 0, tmpl::list<AdditionalArgs...>>::
      template f<
          std::decay_t<std::tuple_element_t<
              0,
              std::decay_t<decltype(Action::apply(
                  std::declval<std::add_lvalue_reference_t<LastReturnType>>(),
                  std::declval<
                      std::add_lvalue_reference_t<AdditionalArgs>>()...))>>>,
          tmpl::push_back<ReturnTypeList, LastReturnType>, Actions...>;
};

/*!
 * \ingroup ParallelGroup
 * \brief Returns a typelist of the return types of all Actions in ActionList
 *
 * \metareturns
 * typelist
 *
 * \tparam ActionsPack parameter pack of Actions taken
 * \tparam FirstInputParameterType the type of the first argument of the first
 * Action in the ActionsPack
 * \tparam AdditionalArgsList the types of the arguments after the first
 * argument, which must all be the same for all Actions in the ActionsPack
 */
template <typename FirstInputParameterType, typename AdditionalArgsList,
          typename... ActionsPack>
using build_action_return_typelist =
    typename Algorithm_detail::build_action_return_types_impl<
        sizeof...(ActionsPack) != 0, AdditionalArgsList>::
        template f<FirstInputParameterType, tmpl::list<>, ActionsPack...>;

CREATE_IS_CALLABLE(is_ready)

CREATE_IS_CALLABLE(apply)

template <typename Invokable, typename ThisVariant, typename... Variants,
          typename... Args,
          Requires<is_apply_callable_v<
              Invokable, std::add_lvalue_reference_t<ThisVariant>, Args&&...>> =
              nullptr>
void apply_visitor_helper(boost::variant<Variants...>& box,
                          const gsl::not_null<int*> iter,
                          const gsl::not_null<bool*> already_visited,
                          Args&&... args) {
  if (box.which() == *iter and not*already_visited) {
    try {
      make_overloader(
          [&box](std::true_type /*returns_void*/, auto&&... my_args) {
            Invokable::apply(boost::get<ThisVariant>(box),
                             std::forward<Args>(my_args)...);
          },
          [&box](std::false_type /*returns_void*/, auto&&... my_args) {
            box = std::get<0>(Invokable::apply(boost::get<ThisVariant>(box),
                                               std::forward<Args>(my_args)...));
          })(
          typename std::is_same<void, decltype(Invokable::apply(
                                          std::declval<ThisVariant&>(),
                                          std::declval<Args>()...))>::type{},
          std::forward<Args>(args)...);

    } catch (std::exception& e) {
      ERROR("Fatal error: Failed to call single Action '"
            << pretty_type::get_name<Invokable>() << "' on iteration '" << iter
            << "' with DataBox type '" << pretty_type::get_name<ThisVariant>()
            << "'\nThe exception is: '" << e.what() << "'\n");
    }
    *already_visited = true;
  }
  (*iter)++;
}

template <typename Invokable, typename ThisVariant, typename... Variants,
          typename... Args,
          Requires<not is_apply_callable_v<
              Invokable, std::add_lvalue_reference_t<ThisVariant>, Args&&...>> =
              nullptr>
void apply_visitor_helper(boost::variant<Variants...>& box,
                          const gsl::not_null<int*> iter,
                          const gsl::not_null<bool*> already_visited,
                          Args&&... /*args*/) {
  if (box.which() == *iter and not*already_visited) {
    ERROR("Cannot call apply function of '"
          << pretty_type::get_name<Invokable>() << "' with DataBox type '"
          << pretty_type::get_name<ThisVariant>() << "' and arguments '"
          << pretty_type::get_name<tmpl::list<Args...>>() << "'");
  }
  (*iter)++;
}

/*!
 * \brief Calls an `Invokable`'s `apply` static member function with the current
 * type in the `boost::variant`.
 *
 * The primary use case for this is to allow executing a single Action at any
 * point in the Algorithm. The current best-known use case for this is setting
 * up initial data. However, the implementation is generic enough to handle a
 * call at any time that is valid. Here valid is defined as the `apply` function
 * only accesses members of the DataBox that are guaranteed to be present when
 * it is invoked, and returns a DataBox of a type that does not break the
 * Algorithm.
 */
template <typename Invokable, typename... Variants, typename... Args>
void apply_visitor(boost::variant<Variants...>& box, Args&&... args) {
  // iter is the current element of the variant in the "for loop"
  int iter = 0;
  // already_visited ensures that only one visitor is invoked
  bool already_visited = false;
  static_cast<void>(std::initializer_list<char>{
      (apply_visitor_helper<Invokable, Variants>(box, &iter, &already_visited,
                                                 std::forward<Args>(args)...),
       '0')...});
}
}  // namespace Algorithm_detail

/// \cond
template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename ActionList, typename ArrayIndex,
          typename InitialDataBox>
class AlgorithmImpl;
/// \endcond

/*!
 * \ingroup ParallelGroup
 * \brief A distributed object (Charm++ Chare) that executes a series of Actions
 * and is capable of sending and receiving data. Acts as an interface to
 * Charm++.
 *
 * ### Different Types of Algorithms
 * Charm++ chares can be one of four types, which is specified by the first
 * template parameter, `ChareType`. The four available types of Algorithms are:
 * 1. A Parallel::Algorithms::Singleton where there is only one
 * in the entire execution of the program.
 * 2. A Parallel::Algorithms::Array which holds zero or more
 * elements each of which is a distributed object on some core. An array can
 * grow and shrink in size dynamically if need be and can also be bound to
 * another array. That is, the bound array has the same number of elements as
 * the array it is bound to, and elements with the same ID are on the same core.
 * 3. A Parallel::Algorithms::Group, which is an array but there is
 * one element per core and they are not able to be moved around between cores.
 * These are typically useful for gathering data from array elements on their
 * core, and then processing or reducing it.
 * 4. A Parallel::Algorithms::Nodegroup, which is similar to a
 * group except that there is one element per node. For Charm++ SMP (shared
 * memory parallelism) builds a node corresponds to the usual definition of a
 * node on a supercomputer. However, for non-SMP builds nodes and cores are
 * equivalent. An important difference between groups and nodegroups is that
 * entry methods (remote calls to functions) are not threadsafe on nodegroups.
 * It is up to the person writing the Actions that will be executed on the
 * Nodegroup Algorithm to ensure they are threadsafe.
 *
 * ### What is an Algorithm?
 * An Algorithm is a distributed object, a Charm++ chare, that repeatedly
 * executes a series of Actions. An Action is a struct that has a `static` apply
 * function with signature:
 *
 * \code
 * template <typename... DbTags, typename... InboxTags, typename Metavariables,
 * typename ArrayIndex,  typename ActionList>
 * static auto  apply(db::DataBox<tmpl::list<DbTags...>>& box,
 *                    tuples::TaggedTuple<InboxTags...>& inboxes,
 *                    const ConstGlobalCache<Metavariables>& cache,
 *                    const ArrayIndex& array_index,
 *                    const TemporalId& temporal_id, const ActionList meta);
 * \endcode
 *
 * Note that any of the arguments can be const or non-const references except
 * `array_index`, which must be a `const&`.
 *
 * ### Explicit instantiations of entry methods
 * The code in src/Parallel/CharmMain.cpp registers all entry methods, and if
 * one is not properly registered then a static_assert explains how to have it
 * be registered. If there is a bug in the implementation and an entry method
 * isn't being registered or hitting a static_assert then Charm++ will give an
 * error of the following form:
 *
 * \verbatim
 * registration happened after init Entry point: explicit_single_action(), addr:
 * 0x555a3d0e2090
 * ------------- Processor 0 Exiting: Called CmiAbort ------------
 * Reason: Did you forget to instantiate a templated entry method in a .ci file?
 * \endverbatim
 *
 * If you encounter this issue please file a bug report supplying everything
 * necessary to reproduce the issue.
 */
template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
class AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
                    tmpl::list<ActionsPack...>, ArrayIndex, InitialDataBox> {
 public:
  /// The metavariables class passed to the Algorithm
  using metavariables = Metavariables;
  /// List of Actions in the order they will be executed
  using actions_list = typelist<ActionsPack...>;
  /// List off all the Tags that can be received into the Inbox
  using inbox_tags_list = Parallel::get_inbox_tags<actions_list>;
  /// The type of the object used to identify the element of the array, group
  /// or nodegroup spatially. The default should be an `int`.
  using array_index = ArrayIndex;

  using parallel_component = ParallelComponent;
  /// The type of the Chare
  using chare_type = ChareType;
  /// The Charm++ proxy object type
  using cproxy_type =
      typename ChareType::template cproxy<ParallelComponent, metavariables,
                                          actions_list, array_index,
                                          InitialDataBox>;
  /// The Charm++ base object type
  using cbase_type =
      typename ChareType::template cbase<ParallelComponent, metavariables,
                                         actions_list, array_index,
                                         InitialDataBox>;
  /// \cond
  // The types held by the boost::variant, box_
  using databox_types = Algorithm_detail::build_action_return_typelist<
      InitialDataBox,
      tmpl::list<tuples::TaggedTupleTypelist<inbox_tags_list>,
                 Parallel::ConstGlobalCache<metavariables>, array_index,
                 actions_list, std::add_pointer_t<ParallelComponent>>,
      ActionsPack...>;
  /// \endcond

  /// \cond
  // Needed for serialization
  AlgorithmImpl() noexcept;
  /// \endcond

  /// Constructor used by Main to initialize the algorithm
  explicit AlgorithmImpl(const Parallel::CProxy_ConstGlobalCache<metavariables>&
                             global_cache_proxy) noexcept;

  /// Charm++ migration constructor, used after a chare is migrated
  constexpr explicit AlgorithmImpl(CkMigrateMessage* /*msg*/) noexcept;

  /// \cond
  ~AlgorithmImpl();

  AlgorithmImpl(const AlgorithmImpl& /*unused*/) = delete;
  AlgorithmImpl& operator=(const AlgorithmImpl& /*unused*/) = delete;
  AlgorithmImpl(AlgorithmImpl&& /*unused*/) = delete;
  AlgorithmImpl& operator=(AlgorithmImpl&& /*unused*/) = delete;
  /// \endcond

  /*!
   * \brief Calls the `apply` function `Action` after a reduction has been
   * completed.
   *
   * The `apply` function must take `arg` as its last argument.
   */
  template <typename Action, typename Arg>
  void reduction_action(Arg arg) noexcept;

  /// \brief Explicitly call the action `Action`. If the returned DataBox type
  /// is not one of the types of the algorithm then a compilation error occurs.
  template <typename Action, typename... Args>
  void explicit_single_action(std::tuple<Args...> args) noexcept;

  template <typename Action>
  void explicit_single_action() noexcept;

  /// Call an Action on a local nodegroup requiring the Action to handle thread
  /// safety.
  ///
  /// The `CmiNodelock` of the nodegroup is passed to the Action instead of the
  /// `action_list` as a `const gsl::not_null<CmiNodelock*>&`. The node lock can
  /// be locked with the `Parallel::lock()` function, and unlocked with
  /// `Parallel::unlock()`. `Parallel::try_lock()` is also provided in case
  /// something useful can be done if the lock couldn't be acquired.
  template <
      typename Action, typename... Args,
      Requires<(sizeof...(Args),
                cpp17::is_same_v<Parallel::Algorithms::Nodegroup, ChareType>)> =
          nullptr>
  void threaded_single_action(Args&&... args) noexcept;

  /// \brief Receive data and store it in the Inbox, and try to continue
  /// executing the algorithm
  ///
  /// When an algorithm has terminated it can be restarted by passing
  /// `enable_if_disabled = true`. This allows long-term disabling and
  /// re-enabling of algorithms
  template <typename ReceiveTag, typename ReceiveDataType>
  void receive_data(typename ReceiveTag::temporal_id instance,
                    ReceiveDataType&& t,
                    bool enable_if_disabled = false) noexcept;

  /// Start evaluating the algorithm until the is_ready function of an Action
  /// returns false, or an Action returns with `terminate` set to `true`
  constexpr void perform_algorithm() noexcept;

  /// Tell the Algorithm it should no longer execute the algorithm. This does
  /// not mean that the execution of the program is terminated, but only that
  /// the algorithm has terminated. An algorithm can be restarted by pass `true`
  /// as the second argument to the `receive_data` method.
  constexpr void set_terminate(bool t) noexcept { terminate_ = t; }

  /// Check if an algorithm should continue being evaluated
  constexpr bool get_terminate() const noexcept { return terminate_; }

 private:
  static constexpr bool is_singleton =
      cpp17::is_same_v<ChareType, Parallel::Algorithms::Singleton>;

  template <class Dummy = int,
            Requires<(sizeof(Dummy), is_singleton)> = nullptr>
  constexpr void set_array_index() noexcept {}
  template <class Dummy = int,
            Requires<(sizeof(Dummy), not is_singleton)> = nullptr>
  void set_array_index() noexcept {
    // down cast to the algorithm_type, so that the `thisIndex` method can be
    // called, which is defined in the CBase class
    array_index_ = static_cast<typename ChareType::template algorithm_type<
        ParallelComponent, Metavariables, tmpl::list<ActionsPack...>,
        ArrayIndex, InitialDataBox>&>(*this)
                       .thisIndex;
  }

  template <size_t... Is>
  constexpr bool iterate_over_actions(
      std::index_sequence<Is...> /*meta*/) noexcept;

  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_action(std::tuple<Args...>&& args,
                               std::index_sequence<Is...> /*meta*/) noexcept {
    Algorithm_detail::apply_visitor<Action>(
        box_, inboxes_, *const_global_cache_,
        static_cast<const array_index&>(array_index_), actions_list{},
        std::add_pointer_t<ParallelComponent>{},
        std::forward<Args>(std::get<Is>(args))...);
  }

  // @{
  /// Since it's not clear how or if it's possible at all to do SFINAE with
  /// Charm's ci files, we use a forward to implementation, where the
  /// implementation is a simple function call that we can use SFINAE with
  template <typename ReceiveTag, typename ReceiveDataType,
            Requires<tt::is_maplike_v<typename ReceiveTag::type::mapped_type>> =
                nullptr>
  void receive_data_impl(typename ReceiveTag::temporal_id& instance,
                         ReceiveDataType&& t);

  template <
      typename ReceiveTag, typename ReceiveDataType,
      Requires<tt::is_a_v<std::unordered_multiset,
                          typename ReceiveTag::type::mapped_type>> = nullptr>
  constexpr void receive_data_impl(typename ReceiveTag::temporal_id& instance,
                                   ReceiveDataType&& t);
  // @}

  // Member variables

#ifdef SPECTRE_CHARM_PROJECTIONS
  double non_action_time_start_;
#endif

  Parallel::ConstGlobalCache<Metavariables>* const_global_cache_{nullptr};
  bool performing_action_ = false;
  std::size_t algorithm_step_ = 0;
  tmpl::conditional_t<Parallel::is_node_group_proxy<cproxy_type>::value,
                      CmiNodeLock, NoSuchType>
      node_lock_;

  bool terminate_{false};
  // Create a boost::variant that can hold any of the DataBox's
  make_boost_variant_over<
      tmpl::append<tmpl::list<db::DataBox<tmpl::list<>>>, databox_types>>
      box_;
  tuples::TaggedTupleTypelist<inbox_tags_list> inboxes_{};
  array_index array_index_;
  // int temporal_id_;
};

////////////////////////////////////////////////////////////////
// Definitions
////////////////////////////////////////////////////////////////

/// \cond
template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
              tmpl::list<ActionsPack...>, ArrayIndex,
              InitialDataBox>::AlgorithmImpl() noexcept {
  make_overloader([](CmiNodeLock& node_lock) { node_lock = CmiCreateLock(); },
                  [](NoSuchType /*unused*/) {})(node_lock_);
  set_array_index();
}

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
              tmpl::list<ActionsPack...>, ArrayIndex, InitialDataBox>::
    AlgorithmImpl(const Parallel::CProxy_ConstGlobalCache<metavariables>&
                      global_cache_proxy) noexcept
    : AlgorithmImpl() {
  const_global_cache_ = global_cache_proxy.ckLocalBranch();
}

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
constexpr AlgorithmImpl<
    ParallelComponent, ChareType, Metavariables, tmpl::list<ActionsPack...>,
    ArrayIndex,
    InitialDataBox>::AlgorithmImpl(CkMigrateMessage* /*msg*/) noexcept
    : AlgorithmImpl() {}

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
              tmpl::list<ActionsPack...>, ArrayIndex,
              InitialDataBox>::~AlgorithmImpl() {
  make_overloader(
      [](CmiNodeLock& node_lock) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
        CmiDestroyLock(node_lock);
#pragma GCC diagnostic pop
      },
      [](NoSuchType /*unused*/) {})(node_lock_);
}

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
template <typename Action, typename Arg>
void AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
                   tmpl::list<ActionsPack...>, ArrayIndex,
                   InitialDataBox>::reduction_action(Arg arg) noexcept {
  lock(&node_lock_);
  static_assert(
      tmpl::found<typename ParallelComponent::reduction_actions_list,
                  std::is_same<tmpl::_1, tmpl::pin<Action>>>::value and
          cpp17::is_same_v<typename Action::reduction_type, std::decay_t<Arg>>,
      "Could not find explicit instantiation of the correct "
      "reduction_action function, which is undefined behavior. See the first "
      "template parameter of 'Parallel::AlgorithmImpl' for which "
      "ParallelComponent is missing the explicit instantiation. The two "
      "reasons this error occurs is missing the Action in the "
      "'reduction_actions_list' of the ParallelComponent, or the Action's "
      "reduction_type member type alias being of the incorrect type.");
  if (performing_action_) {
    ERROR(
        "Already performing an Action and cannot execute additional Actions "
        "from inside of an Action. This is only possible if the "
        "reduction_action function is not invoked via a proxy, which makes "
        "no sense for a reduction.");
  }
  performing_action_ = true;
  Algorithm_detail::apply_visitor<Action>(
      box_, inboxes_, *const_global_cache_,
      static_cast<const array_index&>(array_index_), actions_list{},
      std::add_pointer_t<ParallelComponent>{}, std::forward<Arg>(arg));
  performing_action_ = false;
  unlock(&node_lock_);
}

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
template <typename Action, typename... Args>
void AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
                   tmpl::list<ActionsPack...>, ArrayIndex,
                   InitialDataBox>::explicit_single_action(std::tuple<Args...>
                                                               args) noexcept {
  lock(&node_lock_);
  static_assert(
      tmpl::found<typename ParallelComponent::explicit_single_actions_list,
                  std::is_same<tmpl::_1, tmpl::pin<Action>>>::value and
          cpp17::is_same_v<typename Action::apply_args, tmpl::list<Args...>>,
      "Could not find explicit instantiation of the correct explicit "
      "single action, which is undefined behavior. See the first template "
      "parameter of 'Parallel::AlgorithmImpl' for which ParallelComponent is "
      "missing the explicit instantiation. An example of an "
      "explicit_single_actions_list is: typelist<initialize>");
  if (performing_action_) {
    ERROR(
        "Already performing an Action and cannot execute additional Actions "
        "from inside of an Action. This is only possible if the "
        "explicit_single_action function is not invoked via a proxy, which "
        "we do not allow.");
  }
  performing_action_ = true;
  forward_tuple_to_action<Action>(std::move(args),
                                  std::make_index_sequence<sizeof...(Args)>{});
  performing_action_ = false;
  unlock(&node_lock_);
}

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
template <typename Action>
void AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
                   tmpl::list<ActionsPack...>, ArrayIndex,
                   InitialDataBox>::explicit_single_action() noexcept {
  lock(&node_lock_);
  static_assert(
      tmpl::found<typename ParallelComponent::explicit_single_actions_list,
                  std::is_same<tmpl::_1, tmpl::pin<Action>>>::value,
      "Could not find explicit instantiation of the correct explicit "
      "single action, which is undefined behavior. See the first template "
      "parameter of 'Parallel::AlgorithmImpl' for which ParallelComponent is "
      "missing the explicit instantiation. An example of an "
      "explicit_single_actions_list is: typelist<initialize>");
  if (performing_action_) {
    ERROR(
        "Already performing an Action and cannot execute additional Actions "
        "from inside of an Action. This is only possible if the "
        "explicit_single_action function is not invoked via a proxy, which "
        "we do not allow.");
  }
  performing_action_ = true;
  Algorithm_detail::apply_visitor<Action>(
      box_, inboxes_, *const_global_cache_,
      static_cast<const array_index&>(array_index_), actions_list{},
      std::add_pointer_t<ParallelComponent>{});
  performing_action_ = false;
  unlock(&node_lock_);
}

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
template <
    typename Action, typename... Args,
    Requires<(sizeof...(Args),
              cpp17::is_same_v<Parallel::Algorithms::Nodegroup, ChareType>)>>
void AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
                   tmpl::list<ActionsPack...>, ArrayIndex, InitialDataBox>::
    threaded_single_action(Args&&... args) noexcept {
  const gsl::not_null<CmiNodeLock*> node_lock{&node_lock_};
  Algorithm_detail::apply_visitor<Action>(
      box_, inboxes_, *const_global_cache_,
      static_cast<const array_index&>(array_index_), node_lock,
      std::forward<Args>(args)...);
}

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
template <typename ReceiveTag, typename ReceiveDataType>
void AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
                   tmpl::list<ActionsPack...>, ArrayIndex, InitialDataBox>::
    receive_data(typename ReceiveTag::temporal_id instance, ReceiveDataType&& t,
                 const bool enable_if_disabled) noexcept {
  try {
    lock(&node_lock_);
    if (enable_if_disabled) {
      set_terminate(false);
    }
    receive_data_impl<ReceiveTag>(instance, std::forward<ReceiveDataType>(t));
    unlock(&node_lock_);
  } catch (std::exception& e) {
    ERROR("Fatal error: Unexpected exception caught in receive_data: "
          << e.what());
  }
  perform_algorithm();
}

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
constexpr void AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
                             tmpl::list<ActionsPack...>, ArrayIndex,
                             InitialDataBox>::perform_algorithm() noexcept {
  if (performing_action_ or get_terminate()) {
    return;
  }
#ifdef SPECTRE_CHARM_PROJECTIONS
  non_action_time_start_ = Parallel::wall_time();
#endif
  lock(&node_lock_);
  while (sizeof...(ActionsPack) > 0 and not get_terminate() and
         iterate_over_actions(
             std::make_index_sequence<sizeof...(ActionsPack)>{})) {
  }
  unlock(&node_lock_);
#ifdef SPECTRE_CHARM_PROJECTIONS
  traceUserBracketEvent(SPECTRE_CHARM_NON_ACTION_WALLTIME_EVENT_ID,
                        non_action_time_start_, Parallel::wall_time());
#endif
}
/// \endcond

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
template <size_t... Is>
constexpr bool
AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
              tmpl::list<ActionsPack...>, ArrayIndex, InitialDataBox>::
    iterate_over_actions(const std::index_sequence<Is...> /*meta*/) noexcept {
  bool take_next_action = true;
  const auto helper = [ this, &take_next_action ](auto iteration) noexcept {
    constexpr size_t iter = decltype(iteration)::value;
    if (not(take_next_action and not terminate_ and algorithm_step_ == iter)) {
      return;
    }
    using this_action = tmpl::at_c<actions_list, iter>;
    using this_databox =
        tmpl::at_c<databox_types,
                   iter == 0 ? tmpl::size<databox_types>::value - 1 : iter>;
    this_databox box{};

    try {
      box = boost::get<this_databox>(box_);
    } catch (std::exception& e) {
      ERROR(
          "\nFailed to retrieve Databox in take_next_action:\nCaught "
          "exception: '"
          << e.what() << "'\nDataBox type: '"
          << pretty_type::get_name<this_databox>() << "'\nIteration: " << iter
          << "\nAction: '" << pretty_type::get_name<this_action>() << "'\n\n");
    }

    const auto check_if_ready = make_overloader(
        [this, &box](std::true_type /*has_is_ready*/, auto t) {
          return decltype(t)::is_ready(
              static_cast<const this_databox&>(box),
              static_cast<const tuples::TaggedTupleTypelist<inbox_tags_list>&>(
                  inboxes_),
              *const_global_cache_,
              static_cast<const array_index&>(array_index_));
        },
        [](std::false_type /*has_is_ready*/, auto) { return true; });

    if (not check_if_ready(
            Algorithm_detail::is_is_ready_callable_t<
                this_action, this_databox,
                tuples::TaggedTupleTypelist<inbox_tags_list>,
                Parallel::ConstGlobalCache<Metavariables>, array_index>{},
            this_action{})) {
      take_next_action = false;
      return;
    }

#ifdef SPECTRE_CHARM_PROJECTIONS
    traceUserBracketEvent(SPECTRE_CHARM_NON_ACTION_WALLTIME_EVENT_ID,
                          non_action_time_start_, Parallel::wall_time());
    double start_time = detail::start_trace_action<this_action>();
#endif
    performing_action_ = true;
    algorithm_step_++;
    make_overloader(
        [this](auto& my_box, std::integral_constant<size_t, 1> /*meta*/)
            SPECTRE_JUST_ALWAYS_INLINE noexcept {
              std::tie(box_) = this_action::apply(
                  my_box, inboxes_, *const_global_cache_,
                  static_cast<const array_index&>(array_index_), actions_list{},
                  std::add_pointer_t<ParallelComponent>{});
            },
        [this](auto& my_box, std::integral_constant<size_t, 2> /*meta*/)
            SPECTRE_JUST_ALWAYS_INLINE noexcept {
              std::tie(box_, terminate_) = this_action::apply(
                  my_box, inboxes_, *const_global_cache_,
                  static_cast<const array_index&>(array_index_), actions_list{},
                  std::add_pointer_t<ParallelComponent>{});
            },
        [this](auto& my_box, std::integral_constant<size_t, 3> /*meta*/)
            SPECTRE_JUST_ALWAYS_INLINE noexcept {
              std::tie(box_, terminate_, algorithm_step_) = this_action::apply(
                  my_box, inboxes_, *const_global_cache_,
                  static_cast<const array_index&>(array_index_), actions_list{},
                  std::add_pointer_t<ParallelComponent>{});
            })(
        box, typename std::tuple_size<decltype(this_action::apply(
                 box, inboxes_, *const_global_cache_,
                 static_cast<const array_index&>(array_index_), actions_list{},
                 std::add_pointer_t<ParallelComponent>{}))>::type{});

    performing_action_ = false;
#ifdef SPECTRE_CHARM_PROJECTIONS
    detail::stop_trace_action<this_action>(start_time);
    non_action_time_start_ = Parallel::wall_time();
#endif
    // Wrap counter if necessary
    if (algorithm_step_ >= sizeof...(ActionsPack)) {
      algorithm_step_ = 0;
    }
  };
  // In case of no Actions
  static_cast<void>(helper);
  // This is a template for loop for Is
  static_cast<void>(std::initializer_list<char>{
      (static_cast<void>(helper(std::integral_constant<size_t, Is>{})),
       '0')...});
  return take_next_action;
}

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
template <typename ReceiveTag, typename ReceiveDataType,
          Requires<tt::is_maplike_v<typename ReceiveTag::type::mapped_type>>>
void AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
                   tmpl::list<ActionsPack...>, ArrayIndex, InitialDataBox>::
    receive_data_impl(typename ReceiveTag::temporal_id& instance,
                      ReceiveDataType&& t) {
  static_assert(
      cpp17::is_same_v<
          cpp20::remove_cvref_t<typename ReceiveDataType::first_type>,
          typename ReceiveTag::type::mapped_type::key_type> and
          cpp17::is_same_v<
              cpp20::remove_cvref_t<typename ReceiveDataType::second_type>,
              typename ReceiveTag::type::mapped_type::mapped_type>,
      "The type of the data passed to receive_data for a tag that holds a map "
      "must be a std::pair.");
#ifdef SPECTRE_CHARM_RECEIVE_MAP_DATA_EVENT_ID
  double start_time = Parallel::wall_time();
#endif
  auto& inbox = tuples::get<ReceiveTag>(inboxes_)[instance];
  ASSERT(0 == inbox.count(t.first),
         "Receiving data from the 'same' source twice. The message id is: "
             << t.first);
  if (not inbox.insert(std::forward<ReceiveDataType>(t)).second) {
    ERROR("Failed to insert data to receive at instance '"
          << instance << "' with tag '" << pretty_type::get_name<ReceiveTag>()
          << "'.\n");
  }
#ifdef SPECTRE_CHARM_RECEIVE_MAP_DATA_EVENT_ID
  traceUserBracketEvent(SPECTRE_CHARM_RECEIVE_MAP_DATA_EVENT_ID, start_time,
                        Parallel::wall_time());
#endif
}

template <typename ParallelComponent, typename ChareType,
          typename Metavariables, typename... ActionsPack, typename ArrayIndex,
          typename InitialDataBox>
template <typename ReceiveTag, typename ReceiveDataType,
          Requires<tt::is_a_v<std::unordered_multiset,
                              typename ReceiveTag::type::mapped_type>>>
constexpr void
AlgorithmImpl<ParallelComponent, ChareType, Metavariables,
              tmpl::list<ActionsPack...>, ArrayIndex, InitialDataBox>::
    receive_data_impl(typename ReceiveTag::temporal_id& instance,
                      ReceiveDataType&& t) {
  tuples::get<ReceiveTag>(inboxes_)[instance].insert(
      std::forward<ReceiveDataType>(t));
}
}  // namespace Parallel
