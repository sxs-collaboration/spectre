// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/variant/variant.hpp>
#include <charm++.h>
#include <converse.h>
#include <cstddef>
#include <exception>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <pup.h>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/Algorithms/AlgorithmArrayDeclarations.hpp"
#include "Parallel/Algorithms/AlgorithmGroupDeclarations.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroupDeclarations.hpp"
#include "Parallel/Algorithms/AlgorithmSingletonDeclarations.hpp"
#include "Parallel/CharmRegistration.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Parallel/SimpleActionVisitation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Parallel/TypeTraits.hpp"
#include "Utilities/BoostHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

// IWYU pragma: no_include <array>  // for tuple_size

// IWYU pragma: no_include "Parallel/Algorithm.hpp"  // Include... ourself?

namespace Parallel {
/// \cond
template <typename ParallelComponent, typename PhaseDepActionList>
class AlgorithmImpl;
/// \endcond

namespace Algorithm_detail {
template <typename Metavariables, typename Component, typename = std::void_t<>>
struct has_registration_list : std::false_type {};

template <typename Metavariables, typename Component>
struct has_registration_list<
    Metavariables, Component,
  std::void_t<
    typename Metavariables::template registration_list<Component>::type>>
    : std::true_type {};

template <typename Metavariables, typename Component>
constexpr bool has_registration_list_v =
    has_registration_list<Metavariables, Component>::value;
}  // namespace Algorithm_detail

/*!
 * \ingroup ParallelGroup
 * \brief A distributed object (Charm++ Chare) that executes a series of Actions
 * and is capable of sending and receiving data. Acts as an interface to
 * Charm++.
 *
 * ### Different Types of Algorithms
 * Charm++ chares can be one of four types, which is specified by the type alias
 * `chare_type` inside the `ParallelComponent`. The four available types of
 * Algorithms are:
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
 *                    const GlobalCache<Metavariables>& cache,
 *                    const ArrayIndex& array_index,
 *                    const TemporalId& temporal_id, const ActionList meta);
 * \endcode
 *
 * Note that any of the arguments can be const or non-const references except
 * `array_index`, which must be a `const&`.
 *
 * ### Explicit instantiations of entry methods
 * The code in src/Parallel/CharmMain.tpp registers all entry methods, and if
 * one is not properly registered then a static_assert explains how to have it
 * be registered. If there is a bug in the implementation and an entry method
 * isn't being registered or hitting a static_assert then Charm++ will give an
 * error of the following form:
 *
 * \verbatim
 * registration happened after init Entry point: simple_action(), addr:
 * 0x555a3d0e2090
 * ------------- Processor 0 Exiting: Called CmiAbort ------------
 * Reason: Did you forget to instantiate a templated entry method in a .ci file?
 * \endverbatim
 *
 * If you encounter this issue please file a bug report supplying everything
 * necessary to reproduce the issue.
 */
template <typename ParallelComponent, typename... PhaseDepActionListsPack>
class AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>
    : public ParallelComponent::chare_type::template cbase<
          ParallelComponent,
          typename get_array_index<typename ParallelComponent::chare_type>::
              template f<ParallelComponent>> {
  static_assert(
      sizeof...(PhaseDepActionListsPack) > 0,
      "Must have at least one phase dependent action list "
      "(PhaseActions) in a parallel component. See the first template "
      "parameter of 'AlgorithmImpl' in the error message to see which "
      "component doesn't have any phase dependent action lists.");

 public:
  /// List of Actions in the order that generates the DataBox types
  using all_actions_list = tmpl::flatten<
      tmpl::list<typename PhaseDepActionListsPack::action_list...>>;
  /// The metavariables class passed to the Algorithm
  using metavariables = typename ParallelComponent::metavariables;
  /// List off all the Tags that can be received into the Inbox
  using inbox_tags_list = Parallel::get_inbox_tags<all_actions_list>;
  /// The type of the object used to uniquely identify the element of the array,
  /// group, or nodegroup. The default depends on the component, see
  /// ParallelComponentHelpers.
  using array_index = typename get_array_index<
      typename ParallelComponent::chare_type>::template f<ParallelComponent>;

  using parallel_component = ParallelComponent;
  /// The type of the Chare
  using chare_type = typename parallel_component::chare_type;
  /// The Charm++ proxy object type
  using cproxy_type =
      typename chare_type::template cproxy<parallel_component, array_index>;
  /// The Charm++ base object type
  using cbase_type =
      typename chare_type::template cbase<parallel_component, array_index>;
  /// The type of the phases
  using PhaseType =
      typename tmpl::front<tmpl::list<PhaseDepActionListsPack...>>::phase_type;

  using phase_dependent_action_lists = tmpl::list<PhaseDepActionListsPack...>;

  /// \cond
  // Needed for serialization
  AlgorithmImpl();
  /// \endcond

  /// Constructor used by Main to initialize the algorithm
  template <class... InitializationTags>
  AlgorithmImpl(
      const Parallel::CProxy_GlobalCache<metavariables>& global_cache_proxy,
      tuples::TaggedTuple<InitializationTags...> initialization_items);

  /// Charm++ migration constructor, used after a chare is migrated
  explicit AlgorithmImpl(CkMigrateMessage* /*msg*/);

  /// \cond
  ~AlgorithmImpl() override;

  AlgorithmImpl(const AlgorithmImpl& /*unused*/) = delete;
  AlgorithmImpl& operator=(const AlgorithmImpl& /*unused*/) = delete;
  AlgorithmImpl(AlgorithmImpl&& /*unused*/) = delete;
  AlgorithmImpl& operator=(AlgorithmImpl&& /*unused*/) = delete;
  /// \endcond

  /// Print the expanded type aliases
  std::string print_types() const;

  /// Print the current state of the algorithm
  std::string print_state() const;

  /// Print the current contents of the inboxes
  std::string print_inbox() const;

  /// Print the current contents of the DataBox
  std::string print_databox() const;

  void pup(PUP::er& p) override;  // NOLINT

  /*!
   * \brief Calls the `apply` function `Action` after a reduction has been
   * completed.
   *
   * The `apply` function must take `arg` as its last argument.
   */
  template <typename Action, typename Arg>
  void reduction_action(Arg arg);

  /// \brief Explicitly call the action `Action`. If the returned DataBox type
  /// is not one of the types of the algorithm then a compilation error occurs.
  template <typename Action, typename... Args>
  void simple_action(std::tuple<Args...> args);

  template <typename Action>
  void simple_action();

  /// \brief Call the `Action` sychronously, returning a result without any
  /// parallelization. The action is called immediately and control flow returns
  /// to the caller immediately upon completion.
  ///
  /// \note `Action` must have a type alias `return_type` specifying its return
  /// type. This constraint is to simplify the variant visitation logic for the
  /// \ref DataBoxGroup "DataBox".
  template <typename Action, typename... Args>
  typename Action::return_type local_synchronous_action(Args&&... args);

  /// @{
  /// Call an Action on a local nodegroup requiring the Action to handle thread
  /// safety.
  ///
  /// The `Parallel::NodeLock` of the nodegroup is passed to the Action instead
  /// of the `action_list` as a `const gsl::not_null<Parallel::NodeLock*>&`. The
  /// node lock can be locked with the `Parallel::NodeLock::lock()` function,
  /// and unlocked with `Parallel::unlock()`. `Parallel::NodeLock::try_lock()`
  /// is also provided in case something useful can be done if the lock couldn't
  /// be acquired.
  template <
      typename Action, typename... Args,
      Requires<(sizeof...(Args), std::is_same_v<Parallel::Algorithms::Nodegroup,
                                                chare_type>)> = nullptr>
  void threaded_action(std::tuple<Args...> args) {
    // Note: this method is defined inline because GCC fails to compile when the
    // definition is out of line.
    (void)Parallel::charmxx::RegisterThreadedAction<ParallelComponent, Action,
                                                    Args...>::registrar;
    forward_tuple_to_threaded_action<Action>(
        std::move(args), std::make_index_sequence<sizeof...(Args)>{});
  }

  template <typename Action>
  void threaded_action();
  /// @}

  /// \brief Receive data and store it in the Inbox, and try to continue
  /// executing the algorithm
  ///
  /// When an algorithm has terminated it can be restarted by passing
  /// `enable_if_disabled = true`. This allows long-term disabling and
  /// re-enabling of algorithms
  template <typename ReceiveTag, typename ReceiveDataType>
  void receive_data(typename ReceiveTag::temporal_id instance,
                    ReceiveDataType&& t, bool enable_if_disabled = false);

  /// @{
  /// Start evaluating the algorithm until it is stopped by an action.
  void perform_algorithm();

  void perform_algorithm(const bool restart_if_terminated);
  /// @}

  /// Start execution of the phase-dependent action list in `next_phase`. If
  /// `next_phase` has already been visited, execution will resume at the point
  /// where the previous execution of the same phase left off.
  void start_phase(const PhaseType next_phase);

  /// Tell the Algorithm it should no longer execute the algorithm. This does
  /// not mean that the execution of the program is terminated, but only that
  /// the algorithm has terminated. An algorithm can be restarted by passing
  /// `true` as the second argument to the `receive_data` method or by calling
  /// perform_algorithm(true).
  constexpr void set_terminate(const bool t) { terminate_ = t; }

  /// Check if an algorithm should continue being evaluated
  constexpr bool get_terminate() const { return terminate_; }

  /// @{
  /// Wrappers for charm++ informational functions.

  /// Number of processing elements
  inline int number_of_procs() const { return sys::number_of_procs(); }

  /// %Index of my processing element.
  inline int my_proc() const { return sys::my_proc(); }

  /// Number of nodes.
  inline int number_of_nodes() const { return sys::number_of_nodes(); }

  /// %Index of my node.
  inline int my_node() const { return sys::my_node(); }

  /// Number of processing elements on the given node.
  inline int procs_on_node(const int node_index) const {
    return sys::procs_on_node(node_index);
  }

  /// The local index of my processing element on my node.
  /// This is in the interval 0, ..., procs_on_node(my_node()) - 1.
  inline int my_local_rank() const { return sys::my_local_rank(); }

  /// %Index of first processing element on the given node.
  inline int first_proc_on_node(const int node_index) const {
    return sys::first_proc_on_node(node_index);
  }

  /// %Index of the node for the given processing element.
  inline int node_of(const int proc_index) const {
    return sys::node_of(proc_index);
  }

  /// The local index for the given processing element on its node.
  inline int local_rank_of(const int proc_index) const {
    return sys::local_rank_of(proc_index);
  }
  /// @}

 private:
  template <typename ThisVariant, typename... Variants>
  void touch_global_cache_proxy_in_databox_impl(
      boost::variant<Variants...>& box, const gsl::not_null<int*> iter,
      const gsl::not_null<bool*> already_visited);

  template <typename... Variants>
  void touch_global_cache_proxy_in_databox(boost::variant<Variants...>& box);

  template <typename ThisVariant, typename... Variants, typename... Args>
  void perform_registration_or_deregistration_impl(
      PUP::er& p, const boost::variant<Variants...>& box,
      const gsl::not_null<int*> iter,
      const gsl::not_null<bool*> already_visited);

  template <typename... Variants, typename... Args>
  void perform_registration_or_deregistration(
      PUP::er& p, const boost::variant<Variants...>& box);

  void set_array_index();

  template <typename PhaseDepActions, size_t... Is>
  constexpr bool iterate_over_actions(std::index_sequence<Is...> /*meta*/);

  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_action(std::tuple<Args...>&& args,
                               std::index_sequence<Is...> /*meta*/);

  template <typename Action, typename... Args, size_t... Is>
  void forward_tuple_to_threaded_action(
      std::tuple<Args...>&& args, std::index_sequence<Is...> /*meta*/);

  size_t number_of_actions_in_phase(const PhaseType phase) const;

  // Invoke the static `apply` method of `ThisAction`. The if constexprs are for
  // handling the cases where the `apply` method returns a tuple of one, two,
  // or three elements, in order:
  // 1. A DataBox
  // 2. Either:
  //    2a. A bool determining whether or not to terminate (and potentially move
  //        to the next phase), or
  //    2b. An `AlgorithmExecution` object describing whether to continue,
  //        pause, or halt.
  // 3. An unsigned integer corresponding to which action in the current phase's
  //    algorithm to execute next.
  //
  // Returns whether the action ran successfully, i.e., did not return
  // AlgorithmExecution::Retry.
  template <typename ThisAction, typename ActionList, typename DbTags>
  bool invoke_iterable_action(db::DataBox<DbTags>& my_box);

  // After catching an exception, shutdown the simulation
  [[noreturn]] void initiate_shutdown(const std::exception& exception);

  // Member variables

#ifdef SPECTRE_CHARM_PROJECTIONS
  double non_action_time_start_;
#endif

  Parallel::CProxy_GlobalCache<metavariables> global_cache_proxy_;
  bool performing_action_ = false;
  PhaseType phase_{};
  std::unordered_map<PhaseType, size_t> phase_bookmarks_{};
  std::size_t algorithm_step_ = 0;
  tmpl::conditional_t<Parallel::is_node_group_proxy<cproxy_type>::value,
                      Parallel::NodeLock, NoSuchType>
      node_lock_;

  bool terminate_{true};
  bool halt_algorithm_until_next_phase_{false};

  using all_cache_tags = get_const_global_cache_tags<metavariables>;
  using initial_databox = db::compute_databox_type<tmpl::flatten<tmpl::list<
      Tags::MetavariablesImpl<metavariables>,
      Tags::GlobalCacheProxy<metavariables>,
      typename ParallelComponent::initialization_tags,
      Tags::GlobalCacheImplCompute<metavariables>,
      db::wrap_tags_in<Tags::FromGlobalCache, all_cache_tags>>>>;
  // The types held by the boost::variant, box_
  using databox_phase_types = typename Algorithm_detail::build_databox_types<
      tmpl::list<>, phase_dependent_action_lists, initial_databox,
      inbox_tags_list, metavariables, array_index, ParallelComponent>::type;

  template <typename T>
  struct get_databox_types {
    using type = typename T::databox_types;
  };

  // Make the DataBox types public so `Main` can print some info about it
 public:
  using databox_types = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::transform<databox_phase_types, get_databox_types<tmpl::_1>>>>;

 private:
  // Create a boost::variant that can hold any of the DataBox's
  using variant_boxes =
      tmpl::push_front<databox_types, db::DataBox<tmpl::list<>>>;
  make_boost_variant_over<variant_boxes> box_;
  tuples::tagged_tuple_from_typelist<inbox_tags_list> inboxes_{};
  array_index array_index_;
};

////////////////////////////////////////////////////////////////
// Definitions
////////////////////////////////////////////////////////////////

/// \cond
template <typename ParallelComponent, typename... PhaseDepActionListsPack>
AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    AlgorithmImpl() {
  set_array_index();
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <class... InitializationTags>
AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    AlgorithmImpl(
        const Parallel::CProxy_GlobalCache<metavariables>& global_cache_proxy,
        tuples::TaggedTuple<InitializationTags...>
            initialization_items)
    : AlgorithmImpl() {
  try {
    (void)initialization_items;  // avoid potential compiler warnings if unused
    // When we are using the LoadBalancing phase, we want the Main component to
    // handle the synchronization, so the components do not participate in the
    // charm++ `AtSync` barrier.
    // The array parallel components are migratable so they get balanced
    // appropriately when load balancing is triggered by the LoadBalancing phase
    // in Main
    if constexpr (std::is_same_v<typename ParallelComponent::chare_type,
                                 Parallel::Algorithms::Array> and
                  Algorithm_detail::has_LoadBalancing_v<
                      typename metavariables::Phase>) {
      this->usesAtSync = false;
      this->setMigratable(true);
    }
    global_cache_proxy_ = global_cache_proxy;
    box_ = db::create<
        db::AddSimpleTags<tmpl::flatten<
            tmpl::list<Tags::MetavariablesImpl<metavariables>,
                       Tags::GlobalCacheProxy<metavariables>,
                       typename ParallelComponent::initialization_tags>>>,
        db::AddComputeTags<
            Tags::GlobalCacheImplCompute<metavariables>,
            db::wrap_tags_in<Tags::FromGlobalCache, all_cache_tags>>>(
        metavariables{}, global_cache_proxy_,
        std::move(get<InitializationTags>(initialization_items))...);
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    AlgorithmImpl(CkMigrateMessage* msg)
    : cbase_type(msg) {}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
AlgorithmImpl<ParallelComponent,
              tmpl::list<PhaseDepActionListsPack...>>::~AlgorithmImpl() {
  // We place the registrar in the destructor since every AlgorithmImpl will
  // have a destructor, but we have different constructors so it's not clear
  // which will be instantiated.
  (void)Parallel::charmxx::RegisterParallelComponent<
      ParallelComponent>::registrar;
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
std::string AlgorithmImpl<ParallelComponent,
                          tmpl::list<PhaseDepActionListsPack...>>::print_types()
    const {
  std::ostringstream os;
  os << "Algorithm type aliases:\n";
  os << "using all_actions_list = " << pretty_type::get_name<all_actions_list>()
     << ";\n";

  os << "using metavariables = " << pretty_type::get_name<metavariables>()
     << ";\n";
  os << "using inbox_tags_list = " << pretty_type::get_name<inbox_tags_list>()
     << ";\n";
  os << "using array_index = " << pretty_type::get_name<array_index>() << ";\n";
  os << "using parallel_component = "
     << pretty_type::get_name<parallel_component>() << ";\n";
  os << "using chare_type = " << pretty_type::get_name<chare_type>() << ";\n";
  os << "using cproxy_type = " << pretty_type::get_name<cproxy_type>() << ";\n";
  os << "using cbase_type = " << pretty_type::get_name<cbase_type>() << ";\n";
  os << "using PhaseType = " << pretty_type::get_name<PhaseType>() << ";\n";
  os << "using phase_dependent_action_lists = "
     << pretty_type::get_name<phase_dependent_action_lists>() << ";\n";
  os << "using all_cache_tags = " << pretty_type::get_name<all_cache_tags>()
     << ";\n";
  os << "using initial_databox = " << pretty_type::get_name<initial_databox>()
     << ";\n";
  os << "using databox_phase_types = "
     << pretty_type::get_name<databox_phase_types>() << ";\n";
  os << "using databox_types = " << pretty_type::get_name<databox_types>()
     << ";\n";
  os << "using variant_boxes = " << pretty_type::get_name<variant_boxes>()
     << ";\n";
  return os.str();
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
std::string AlgorithmImpl<ParallelComponent,
                          tmpl::list<PhaseDepActionListsPack...>>::print_state()
    const {
  std::ostringstream os;
  os << "State:\n";
  os << "performing_action_ = " << std::boolalpha << performing_action_
     << ";\n";
  os << "phase_ = " << phase_ << ";\n";
  os << "phase_bookmarks_ = " << phase_bookmarks_ << ";\n";
  os << "algorithm_step_ = " << algorithm_step_ << ";\n";
  os << "terminate_ = " << terminate_ << ";\n";
  os << "halt_algorithm_until_next_phase_ = "
     << halt_algorithm_until_next_phase_ << ";\n";
  os << "array_index_ = " << array_index_ << ";\n";
  return os.str();
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
std::string AlgorithmImpl<ParallelComponent,
                          tmpl::list<PhaseDepActionListsPack...>>::print_inbox()
    const {
  std::ostringstream os;
  os << "inboxes_ = " << inboxes_ << ";\n";
  return os.str();
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
std::string
AlgorithmImpl<ParallelComponent,
              tmpl::list<PhaseDepActionListsPack...>>::print_databox() const {
  std::ostringstream os;
  os << "box_:\n" << box_;
  return os.str();
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    pup(PUP::er& p) {  // NOLINT
#ifdef SPECTRE_CHARM_PROJECTIONS
  p | non_action_time_start_;
#endif
  if (performing_action_ and not p.isSizing()) {
    ERROR("cannot serialize while performing action!");
  }
  p | performing_action_;
  p | phase_;
  p | phase_bookmarks_;
  p | algorithm_step_;
  if constexpr (Parallel::is_node_group_proxy<cproxy_type>::value) {
    p | node_lock_;
  }
  p | terminate_;
  p | halt_algorithm_until_next_phase_;
  p | box_;
  // After unpacking the DataBox, we "touch" the GlobalCache proxy inside.
  // This forces the DataBox to recompute the GlobalCache* the next time it
  // is needed, but delays this process until after the pupper is called.
  // (This delay is important: updating the pointer requires calling
  // ckLocalBranch() on the Charm++ proxy, and in a restart from checkpoint
  // this call may not be well-defined until after components are finished
  // unpacking.)
  if (p.isUnpacking()) {
    touch_global_cache_proxy_in_databox(box_);
  }
  p | inboxes_;
  p | array_index_;
  p | global_cache_proxy_;
  // Note that `perform_registration_or_deregistration` passes the `box_` by
  // const reference. If mutable access is required to the box, this function
  // call needs to be carefully considered with respect to the `p | box_` call
  // in both packing and unpacking scenarios.
  //
  // Note also that we don't perform (de)registrations when pup'ing for a
  // checkpoint/restart. This enables a simpler first-pass implementation of
  // checkpointing, though it means the restart must occur on the same
  // hardware configuration (same number of nodes and same procs per node)
  // used when writing the checkpoint.
  if constexpr (Algorithm_detail::has_LoadBalancing_v<
                    typename metavariables::Phase>) {
    if (phase_ == metavariables::Phase::LoadBalancing) {
      perform_registration_or_deregistration(p, box_);
    }
  }
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename Action, typename Arg>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    reduction_action(Arg arg) {
  try {
    (void)Parallel::charmxx::RegisterReductionAction<
        ParallelComponent, Action, std::decay_t<Arg>>::registrar;
    if constexpr (std::is_same_v<Parallel::NodeLock, decltype(node_lock_)>) {
      node_lock_.lock();
    }
    if (performing_action_) {
      ERROR(
          "Already performing an Action and cannot execute additional Actions "
          "from inside of an Action. This is only possible if the "
          "reduction_action function is not invoked via a proxy, which makes "
          "no sense for a reduction.");
    }
    performing_action_ = true;
    arg.finalize();
    forward_tuple_to_action<Action>(
        std::move(arg.data()), std::make_index_sequence<Arg::pack_size()>{});
    performing_action_ = false;
    if constexpr (std::is_same_v<Parallel::NodeLock, decltype(node_lock_)>) {
      node_lock_.unlock();
    }
    perform_algorithm();
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename Action, typename... Args>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    simple_action(std::tuple<Args...> args) {
  try {
    (void)Parallel::charmxx::RegisterSimpleAction<ParallelComponent, Action,
                                                  Args...>::registrar;
    if constexpr (std::is_same_v<Parallel::NodeLock, decltype(node_lock_)>) {
      node_lock_.lock();
    }
    if (performing_action_) {
      ERROR(
          "Already performing an Action and cannot execute additional Actions "
          "from inside of an Action. This is only possible if the "
          "simple_action function is not invoked via a proxy, which "
          "we do not allow.");
    }
    performing_action_ = true;
    forward_tuple_to_action<Action>(
        std::move(args), std::make_index_sequence<sizeof...(Args)>{});
    performing_action_ = false;
    if constexpr (std::is_same_v<Parallel::NodeLock, decltype(node_lock_)>) {
      node_lock_.unlock();
    }
    perform_algorithm();
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename Action>
void AlgorithmImpl<ParallelComponent,
                   tmpl::list<PhaseDepActionListsPack...>>::simple_action() {
  try {
    (void)Parallel::charmxx::RegisterSimpleAction<ParallelComponent,
                                                  Action>::registrar;
    if constexpr (std::is_same_v<Parallel::NodeLock, decltype(node_lock_)>) {
      node_lock_.lock();
    }
    if (performing_action_) {
      ERROR(
          "Already performing an Action and cannot execute additional Actions "
          "from inside of an Action. This is only possible if the "
          "simple_action function is not invoked via a proxy, which "
          "we do not allow.");
    }
    performing_action_ = true;
    Algorithm_detail::simple_action_visitor<Action, ParallelComponent>(
        box_, *Parallel::local_branch(global_cache_proxy_),
        static_cast<const array_index&>(array_index_));
    performing_action_ = false;
    if constexpr (std::is_same_v<Parallel::NodeLock, decltype(node_lock_)>) {
      node_lock_.unlock();
    }
    perform_algorithm();
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename Action, typename... Args>
typename Action::return_type
AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    local_synchronous_action(Args&&... args) {
  static_assert(Parallel::is_node_group_proxy<cproxy_type>::value,
                "Cannot call a (blocking) local synchronous action on a "
                "chare that is not a NodeGroup");
  return Algorithm_detail::local_synchronous_action_visitor<Action,
                                                            ParallelComponent>(
      box_, make_not_null(&node_lock_), std::forward<Args>(args)...);
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename Action>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    threaded_action() {
  try {
    // NOLINTNEXTLINE(modernize-redundant-void-arg)
    (void)Parallel::charmxx::RegisterThreadedAction<ParallelComponent,
                                                    Action>::registrar;
    Algorithm_detail::simple_action_visitor<Action, ParallelComponent>(
        box_, *Parallel::local_branch(global_cache_proxy_),
        static_cast<const array_index&>(array_index_),
        make_not_null(&node_lock_));
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename ReceiveTag, typename ReceiveDataType>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    receive_data(typename ReceiveTag::temporal_id instance, ReceiveDataType&& t,
                 const bool enable_if_disabled) {
  try {
    (void)Parallel::charmxx::RegisterReceiveData<ParallelComponent,
                                                 ReceiveTag>::registrar;
    if constexpr (std::is_same_v<Parallel::NodeLock, decltype(node_lock_)>) {
      node_lock_.lock();
    }
    if (enable_if_disabled) {
      set_terminate(false);
    }
    ReceiveTag::insert_into_inbox(
        make_not_null(&tuples::get<ReceiveTag>(inboxes_)), instance,
        std::forward<ReceiveDataType>(t));
    if constexpr (std::is_same_v<Parallel::NodeLock, decltype(node_lock_)>) {
      node_lock_.unlock();
    }
    perform_algorithm();
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    perform_algorithm() {
  try {
    if (performing_action_ or get_terminate() or
        halt_algorithm_until_next_phase_) {
      return;
    }
#ifdef SPECTRE_CHARM_PROJECTIONS
    non_action_time_start_ = sys::wall_time();
#endif
    if constexpr (std::is_same_v<Parallel::NodeLock, decltype(node_lock_)>) {
      node_lock_.lock();
    }
    const auto invoke_for_phase = [this](auto phase_dep_v) {
      using PhaseDep = decltype(phase_dep_v);
      constexpr PhaseType phase = PhaseDep::phase;
      using actions_list = typename PhaseDep::action_list;
      if (phase_ == phase) {
        while (
            tmpl::size<actions_list>::value > 0 and not get_terminate() and
            not halt_algorithm_until_next_phase_ and
            iterate_over_actions<PhaseDep>(
                std::make_index_sequence<tmpl::size<actions_list>::value>{})) {
        }
      }
    };
    // Loop over all phases, once the current phase is found we perform the
    // algorithm in that phase until we are no longer able to because we are
    // waiting on data to be sent or because the algorithm has been marked as
    // terminated.
    EXPAND_PACK_LEFT_TO_RIGHT(invoke_for_phase(PhaseDepActionListsPack{}));
    if constexpr (std::is_same_v<Parallel::NodeLock, decltype(node_lock_)>) {
      node_lock_.unlock();
    }
#ifdef SPECTRE_CHARM_PROJECTIONS
    traceUserBracketEvent(SPECTRE_CHARM_NON_ACTION_WALLTIME_EVENT_ID,
                          non_action_time_start_, sys::wall_time());
#endif
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    perform_algorithm(const bool restart_if_terminated) {
  try {
    if (restart_if_terminated) {
      set_terminate(false);
    }
    perform_algorithm();
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    start_phase(const PhaseType next_phase) {
  try {
    // terminate should be true since we exited a phase previously.
    if (not get_terminate() and not halt_algorithm_until_next_phase_) {
      ERROR(
          "An algorithm must always be set to terminate at the beginning of a "
          "phase. Since this is not the case the previous phase did not end "
          "correctly. The integer corresponding to the previous phase is: "
          << static_cast<int>(phase_)
          << " and the next phase is: " << static_cast<int>(next_phase)
          << ", The termination flag is: " << get_terminate()
          << ", and the halt flag is: " << halt_algorithm_until_next_phase_);
    }
    // set terminate to true if there are no actions in this PDAL
    set_terminate(number_of_actions_in_phase(next_phase) == 0);

    // Ideally, we'd set the bookmarks as we are leaving a phase, but there is
    // no 'clean-up' code that we run when departing a phase, so instead we set
    // the bookmark for the previous phase (still stored in `phase_` at this
    // point), before we update the member variable `phase_`.
    // Then, after updating `phase_`, we check if we've ever stored a bookmark
    // for the new phase previously. If so, we start from where we left off,
    // otherwise, start from the beginning of the action list.
    phase_bookmarks_[phase_] = algorithm_step_;
    phase_ = next_phase;
    if (phase_bookmarks_.count(phase_) != 0) {
      algorithm_step_ = phase_bookmarks_.at(phase_);
    } else {
      algorithm_step_ = 0;
    }
    halt_algorithm_until_next_phase_ = false;
    perform_algorithm();
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename ThisVariant, typename... Variants>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    touch_global_cache_proxy_in_databox_impl(
        boost::variant<Variants...>& box, const gsl::not_null<int*> iter,
        const gsl::not_null<bool*> already_visited) {
  if constexpr (db::tag_is_retrievable_v<Tags::GlobalCacheProxy<metavariables>,
                                         ThisVariant>) {
    if (box.which() == *iter and not *already_visited) {
      db::mutate<Tags::GlobalCacheProxy<metavariables>>(
          make_not_null(&(boost::get<ThisVariant>(box))),
          [](const gsl::not_null<CProxy_GlobalCache<metavariables>*>
                 proxy) { (void)proxy; });
      *already_visited = true;
    }
  } else {
    // silence warnings
    (void)already_visited;
  }
  ++(*iter);
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename... Variants>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    touch_global_cache_proxy_in_databox(
        boost::variant<Variants...>& box) {
  int iter = 0;
  bool already_visited = false;
  EXPAND_PACK_LEFT_TO_RIGHT(touch_global_cache_proxy_in_databox_impl<Variants>(
      box, &iter, &already_visited));
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename ThisVariant, typename... Variants, typename... Args>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    perform_registration_or_deregistration_impl(
        PUP::er& p, const boost::variant<Variants...>& box,
        const gsl::not_null<int*> iter,
        const gsl::not_null<bool*> already_visited) {
  // void cast to avoid compiler warnings about the unused variable in the
  // false branch of the constexpr
  (void)already_visited;
  if (box.which() == *iter and not *already_visited) {
    // The deregistration and registration below does not actually insert
    // anything into the PUP::er stream, so nothing is done on a sizing pup.
    if constexpr (Algorithm_detail::has_registration_list_v<
                      metavariables, ParallelComponent>) {
      using registration_list =
          typename metavariables::template registration_list<
              ParallelComponent>::type;
      if (p.isPacking()) {
        tmpl::for_each<registration_list>(
            [this, &box](auto registration_v) {
              using registration = typename decltype(registration_v)::type;
              registration::template perform_deregistration<ParallelComponent>(
                  boost::get<ThisVariant>(box),
                  *Parallel::local_branch(global_cache_proxy_), array_index_);
            });
      }
      if (p.isUnpacking()) {
        tmpl::for_each<registration_list>(
            [this, &box](auto registration_v) {
              using registration = typename decltype(registration_v)::type;
              registration::template perform_registration<ParallelComponent>(
                  boost::get<ThisVariant>(box),
                  *Parallel::local_branch(global_cache_proxy_), array_index_);
            });
      }
      *already_visited = true;
    }
  }
  ++(*iter);
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename... Variants, typename... Args>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    perform_registration_or_deregistration(
        PUP::er& p, const boost::variant<Variants...>& box) {
  int iter = 0;
  bool already_visited = false;
  EXPAND_PACK_LEFT_TO_RIGHT(
      perform_registration_or_deregistration_impl<Variants>(p, box, &iter,
                                                            &already_visited));
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    set_array_index() {
  // down cast to the algorithm_type, so that the `thisIndex` method can be
  // called, which is defined in the CBase class
  array_index_ = static_cast<typename chare_type::template algorithm_type<
      ParallelComponent, array_index>&>(*this)
                     .thisIndex;
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename PhaseDepActions, size_t... Is>
constexpr bool
AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    iterate_over_actions(const std::index_sequence<Is...> /*meta*/) {
  bool take_next_action = true;
  const auto helper = [this, &take_next_action](auto iteration) {
    constexpr size_t iter = decltype(iteration)::value;
    if (not(take_next_action and not terminate_ and
            not halt_algorithm_until_next_phase_ and algorithm_step_ == iter)) {
      return;
    }
    using actions_list = typename PhaseDepActions::action_list;
    using this_action = tmpl::at_c<actions_list, iter>;

    constexpr size_t phase_index =
        tmpl::index_of<phase_dependent_action_lists, PhaseDepActions>::value;
    using databox_phase_type = tmpl::at_c<databox_phase_types, phase_index>;
    using databox_types_this_phase = typename databox_phase_type::databox_types;

    using potential_databox_indices = std::conditional_t<
        iter == 0_st,
        tmpl::integral_list<size_t, 0_st,
                            tmpl::size<databox_types_this_phase>::value - 1_st>,
        tmpl::integral_list<size_t, iter>>;
    bool box_found = false;
    tmpl::for_each<potential_databox_indices>(
        [this, &box_found, &take_next_action](auto potential_databox_index_v) {
          constexpr size_t potential_databox_index =
              decltype(potential_databox_index_v)::type::value;
          using this_databox =
              tmpl::at_c<databox_types_this_phase, potential_databox_index>;
          if (not box_found and
              box_.which() ==
                  static_cast<int>(
                      tmpl::index_of<variant_boxes, this_databox>::value)) {
            box_found = true;
            auto& box = boost::get<this_databox>(box_);
            performing_action_ = true;
            ++algorithm_step_;
            if (not invoke_iterable_action<this_action, actions_list>(box)) {
              take_next_action = false;
              --algorithm_step_;
            }
          }
        });
    if (not box_found) {
      ERROR(
          "The DataBox type being retrieved at algorithm step: "
          << algorithm_step_ << " in phase " << phase_index
          << " corresponding to action " << pretty_type::get_name<this_action>()
          << " is not the correct type but is of variant index " << box_.which()
          << ". If you are using Goto and Label actions then you are using "
             "them incorrectly.");
    }

    performing_action_ = false;
    // Wrap counter if necessary
    if (algorithm_step_ >= tmpl::size<actions_list>::value) {
      algorithm_step_ = 0;
    }
  };
  // In case of no Actions avoid compiler warning.
  (void)helper;
  // This is a template for loop for Is
  EXPAND_PACK_LEFT_TO_RIGHT(helper(std::integral_constant<size_t, Is>{}));
  return take_next_action;
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename Action, typename... Args, size_t... Is>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    forward_tuple_to_action(std::tuple<Args...>&& args,
                            std::index_sequence<Is...> /*meta*/) {
  Algorithm_detail::simple_action_visitor<Action, ParallelComponent>(
      box_, *Parallel::local_branch(global_cache_proxy_),
      static_cast<const array_index&>(array_index_),
      std::forward<Args>(std::get<Is>(args))...);
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename Action, typename... Args, size_t... Is>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    forward_tuple_to_threaded_action(
        std::tuple<Args...>&& args,
        std::index_sequence<Is...> /*meta*/) {
  const gsl::not_null<Parallel::NodeLock*> node_lock{&node_lock_};
  Algorithm_detail::simple_action_visitor<Action, ParallelComponent>(
      box_, *Parallel::local_branch(global_cache_proxy_),
      static_cast<const array_index&>(array_index_), node_lock,
      std::forward<Args>(std::get<Is>(args))...);
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
size_t
AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    number_of_actions_in_phase(const PhaseType phase) const {
  size_t number_of_actions = 0;
  const auto helper = [&number_of_actions, phase](auto pdal_v) {
    if (pdal_v.phase == phase) {
      number_of_actions = pdal_v.number_of_actions;
    }
  };
  EXPAND_PACK_LEFT_TO_RIGHT(helper(PhaseDepActionListsPack{}));
  return number_of_actions;
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
template <typename ThisAction, typename ActionList, typename DbTags>
bool AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    invoke_iterable_action(db::DataBox<DbTags>& my_box) {
  static_assert(not Algorithm_detail::is_is_ready_callable_t<
                    ThisAction, db::DataBox<DbTags>&,
                    tuples::tagged_tuple_from_typelist<inbox_tags_list>&,
                    Parallel::GlobalCache<metavariables>&, array_index>{},
                "Actions no longer support is_ready methods.  Instead, "
                "return AlgorithmExecution::Retry from apply().");

  auto action_return = ThisAction::apply(
      my_box, inboxes_, *Parallel::local_branch(global_cache_proxy_),
      std::as_const(array_index_), ActionList{},
      std::add_pointer_t<ParallelComponent>{});

  static_assert(
      Algorithm_detail::check_iterable_action_return_type<
          ParallelComponent, ThisAction,
          std::decay_t<decltype(action_return)>>::value,
      "An iterable action has an invalid return type.\n"
      "See the template parameters of "
      "Algorithm_detail::check_iterable_action_return_type for details: the "
      "first is the parallel component in question, the second is the "
      "iterable action, and the third is the return type at fault.\n"
      "The return type must be a tuple of length one, two, or three "
      "with:\n"
      " first type is an updated DataBox;\n"
      " second type is either a bool (indicating termination) or a "
      "`Parallel::AlgorithmExecution` object;\n"
      " third type is a size_t indicating the next action in the current"
      " phase.");

  constexpr size_t tuple_size = std::tuple_size<decltype(action_return)>::value;
  if constexpr (tuple_size >= 1_st) {
    box_ = std::move(get<0>(action_return));
  }
  if constexpr (tuple_size >= 2_st) {
    if constexpr (std::is_same_v<decltype(get<1>(action_return)), bool&>) {
      terminate_ = get<1>(action_return);
    } else {
      switch (get<1>(action_return)) {
        case AlgorithmExecution::Halt:
          halt_algorithm_until_next_phase_ = true;
          terminate_ = true;
          break;
        case AlgorithmExecution::Pause:
          terminate_ = true;
          break;
        case AlgorithmExecution::Retry:
          if constexpr (tuple_size >= 3_st) {
            ASSERT(get<2>(action_return) == std::numeric_limits<size_t>::max(),
                   "Switching actions on a Retry doesn't make sense.  If you "
                   "need to return a three-element tuple, pass "
                   "std::numeric_limits<size_t>::max() as the last element.");
          }
          return false;
        default:
          break;
      }
    }
  }
  if constexpr (tuple_size >= 3_st) {
    algorithm_step_ = get<2>(action_return);
  }
  return true;
}

template <typename ParallelComponent, typename... PhaseDepActionListsPack>
void AlgorithmImpl<ParallelComponent, tmpl::list<PhaseDepActionListsPack...>>::
    initiate_shutdown(const std::exception& exception) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  CkError("\nException caught in Algorithm:\n%s\n\nShutting down...\n",
          exception.what());
  sys::abort("");
}
/// \endcond

template <typename ParallelComponent, typename PhaseDepActionLists>
std::ostream& operator<<(
    std::ostream& os,
    const AlgorithmImpl<ParallelComponent, PhaseDepActionLists>&
        algorithm_impl) {
  os << algorithm_impl.print_types() << "\n";
  os << algorithm_impl.print_state() << "\n";
  os << algorithm_impl.print_inbox() << "\n";
  os << algorithm_impl.print_databox() << "\n";
  return os;
}
}  // namespace Parallel
