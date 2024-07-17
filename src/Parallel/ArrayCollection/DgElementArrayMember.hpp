// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <cstddef>
#include <exception>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/ArrayCollection/DgElementArrayMemberBase.hpp"
#include "Parallel/ArrayCollection/SetTerminateOnElement.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Tags/ArrayIndex.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/System/Abort.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <size_t Dim, class Metavariables, class PhaseDepActionList>
struct DgElementCollection;
}  // namespace Parallel
/// \endcond

namespace Parallel {
/*!
 * \brief An element or multiple elements stored contiguously on a group or
 * nodegroup.
 *
 * Consider first the simpler case where each `DgElementArrayMember` is single
 * DG or FD element. Each has a DataBox, and inbox. The various bookkeeping
 * constructs are stored in the `DgElementArrayMemberBase`. The
 * `DgElementArrayMember` is effectively the distributed object that has
 * remote entry methods invoked on it. However, a `DgElementArrayMember` is
 * not tied to a particalur core since it lives on a nodegroup (the
 * `DgElementCollection`). It is also possible to use a group instead of a
 * nodegroup, but that is mostly of interest when using GPUs.
 */
template <size_t Dim, typename Metavariables, typename PhaseDepActionList,
          typename SimpleTagsFromOptions>
class DgElementArrayMember;

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class DgElementArrayMember<Dim, Metavariables,
                           tmpl::list<PhaseDepActionListsPack...>,
                           SimpleTagsFromOptions>
    : public DgElementArrayMemberBase<Dim> {
 public:
  using ParallelComponent =
      DgElementCollection<Dim, Metavariables,
                          tmpl::list<PhaseDepActionListsPack...>>;

  /// List of Actions in the order that generates the DataBox types
  using all_actions_list = tmpl::flatten<
      tmpl::list<typename PhaseDepActionListsPack::action_list...>>;
  /// The metavariables class passed to the Algorithm
  using metavariables = Metavariables;
  /// List of all the Tags that can be received into the Inbox
  using inbox_tags_list = Parallel::get_inbox_tags<all_actions_list>;
  using phase_dependent_action_lists = tmpl::list<PhaseDepActionListsPack...>;
  using all_cache_tags = get_const_global_cache_tags<metavariables>;

  using databox_type = db::compute_databox_type<tmpl::flatten<tmpl::list<
      Tags::MetavariablesImpl<metavariables>,
      Tags::ArrayIndexImpl<ElementId<Dim>>,
      Tags::GlobalCacheProxy<metavariables>, SimpleTagsFromOptions,
      Tags::GlobalCacheImplCompute<metavariables>,
      Tags::ResourceInfoReference<metavariables>,
      db::wrap_tags_in<Tags::FromGlobalCache, all_cache_tags>,
      Algorithm_detail::get_pdal_simple_tags<phase_dependent_action_lists>,
      Algorithm_detail::get_pdal_compute_tags<phase_dependent_action_lists>>>>;

  using inbox_type = tuples::tagged_tuple_from_typelist<inbox_tags_list>;

  DgElementArrayMember() = default;

  template <class... InitializationTags>
  DgElementArrayMember(
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache_proxy,
      tuples::TaggedTuple<InitializationTags...> initialization_items,
      ElementId<Dim> element_id);

  /// \cond
  ~DgElementArrayMember() override = default;

  DgElementArrayMember(const DgElementArrayMember& /*unused*/) = default;
  DgElementArrayMember& operator=(const DgElementArrayMember& /*unused*/) =
      default;
  DgElementArrayMember(DgElementArrayMember&& /*unused*/) = default;
  DgElementArrayMember& operator=(DgElementArrayMember&& /*unused*/) = default;

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(DgElementArrayMemberBase<Dim>), DgElementArrayMember);

  explicit DgElementArrayMember(CkMigrateMessage* msg);
  /// \endcond

  /// Start execution of the phase-dependent action list in `next_phase`. If
  /// `next_phase` has already been visited, execution will resume at the point
  /// where the previous execution of the same phase left off.
  void start_phase(Parallel::Phase next_phase) override;

  const auto& databox() const { return box_; }

  /// Start evaluating the algorithm until it is stopped by an action.
  void perform_algorithm() override;

  template <typename ThisAction, typename PhaseIndex, typename DataBoxIndex>
  bool invoke_iterable_action();

  /*!
   * \brief Invokes a simple action on the element.
   *
   * \note This does not lock the element. It is up to the calling action to
   * lock the element if needed.
   */
  template <typename Action, typename... Args>
  void simple_action(Args&&... args) {
    try {
      if (this->performing_action_) {
        ERROR(
            "Already performing an Action and cannot execute additional "
            "Actions from inside of an Action. This is only possible if the "
            "simple_action function is not invoked via a proxy, which "
            "we do not allow.");
      }
      this->performing_action_ = true;
      Action::template apply<ParallelComponent>(
          box_, *Parallel::local_branch(global_cache_proxy_), this->element_id_,
          std::forward<Args>(args)...);
      this->performing_action_ = false;
      perform_algorithm();
    } catch (const std::exception& exception) {
      initiate_shutdown(exception);
    }
  }

  /// Print the expanded type aliases
  std::string print_types() const override;

  /// Print the current contents of the inboxes
  std::string print_inbox() const override;

  /// Print the current contents of the DataBox
  std::string print_databox() const override;

  /// @{
  /// Get read access to all the inboxes
  auto& inboxes() { return inboxes_; }
  const auto& inboxes() const { return inboxes_; }
  const auto& get_inboxes() const { return inboxes(); }
  /// @}

  void pup(PUP::er& p) override;

 private:
  size_t number_of_actions_in_phase(Parallel::Phase phase) const;

  // After catching an exception, shutdown the simulation
  void initiate_shutdown(const std::exception& exception);

  template <typename PhaseDepActions, size_t... Is>
  bool iterate_over_actions(std::index_sequence<Is...> /*meta*/);
  static_assert(std::is_move_constructible_v<databox_type>);
  static_assert(std::is_move_constructible_v<inbox_type>);

  Parallel::CProxy_GlobalCache<Metavariables> global_cache_proxy_{};
  databox_type box_{};
  inbox_type inboxes_{};
};

/// \cond
template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
DgElementArrayMember<
    Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
    SimpleTagsFromOptions>::DgElementArrayMember(CkMigrateMessage* msg)
    : DgElementArrayMemberBase<Dim>(msg) {}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
template <class... InitializationTags>
DgElementArrayMember<Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
                     SimpleTagsFromOptions>::
    DgElementArrayMember(
        const Parallel::CProxy_GlobalCache<Metavariables>& global_cache_proxy,
        tuples::TaggedTuple<InitializationTags...> initialization_items,
        ElementId<Dim> element_id)
    : Parallel::DgElementArrayMemberBase<Dim>(
          std::move(element_id),
          Parallel::my_node<size_t>(
              *Parallel::local_branch(global_cache_proxy))),
      global_cache_proxy_(global_cache_proxy) {
  (void)initialization_items;  // avoid potential compiler warnings if unused
  ::Initialization::mutate_assign<
      tmpl::list<Tags::ArrayIndex, Tags::GlobalCacheProxy<Metavariables>,
                 InitializationTags...>>(
      make_not_null(&box_), this->element_id_, global_cache_proxy_,
      std::move(get<InitializationTags>(initialization_items))...);
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
void DgElementArrayMember<
    Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
    SimpleTagsFromOptions>::start_phase(const Parallel::Phase next_phase) {
  try {
    // terminate should be true since we exited a phase previously.
    if (not this->get_terminate() and
        not this->halt_algorithm_until_next_phase_) {
      ERROR(
          "An algorithm must always be set to terminate at the beginning of a "
          "phase. Since this is not the case the previous phase did not end "
          "correctly. The previous phase is: "
          << this->phase_ << " and the next phase is: " << next_phase
          << ", The termination flag is: " << this->get_terminate()
          << ", and the halt flag is: "
          << this->halt_algorithm_until_next_phase_ << ' '
          << this->element_id_);
    }
    // set terminate to true if there are no actions in this PDAL
    auto& cache = *Parallel::local_branch(global_cache_proxy_);
    Parallel::local_synchronous_action<
        Parallel::Actions::SetTerminateOnElement>(
        Parallel::get_parallel_component<ParallelComponent>(cache),
        make_not_null(&cache), this->element_id_,
        number_of_actions_in_phase(next_phase) == 0);

    // Ideally, we'd set the bookmarks as we are leaving a phase, but there is
    // no 'clean-up' code that we run when departing a phase, so instead we set
    // the bookmark for the previous phase (still stored in `phase_` at this
    // point), before we update the member variable `phase_`.
    // Then, after updating `phase_`, we check if we've ever stored a bookmark
    // for the new phase previously. If so, we start from where we left off,
    // otherwise, start from the beginning of the action list.
    this->phase_bookmarks_[this->phase_] = this->algorithm_step_;
    this->phase_ = next_phase;
    if (this->phase_bookmarks_.count(this->phase_) != 0) {
      this->algorithm_step_ = this->phase_bookmarks_.at(this->phase_);
    } else {
      this->algorithm_step_ = 0;
    }
    this->halt_algorithm_until_next_phase_ = false;
    perform_algorithm();
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
void DgElementArrayMember<Dim, Metavariables,
                          tmpl::list<PhaseDepActionListsPack...>,
                          SimpleTagsFromOptions>::perform_algorithm() {
  try {
    if (this->performing_action_ or this->get_terminate() or
        this->halt_algorithm_until_next_phase_) {
      return;
    }
    const auto invoke_for_phase = [this](auto phase_dep_v) {
      using PhaseDep = decltype(phase_dep_v);
      constexpr Parallel::Phase phase = PhaseDep::phase;
      using actions_list = typename PhaseDep::action_list;
      if (this->phase_ == phase) {
        while (
            tmpl::size<actions_list>::value > 0 and
            not this->get_terminate() and
            not this->halt_algorithm_until_next_phase_ and
            iterate_over_actions<PhaseDep>(
                std::make_index_sequence<tmpl::size<actions_list>::value>{})) {
        }
        tmpl::for_each<actions_list>([this](auto action_v) {
          using action = tmpl::type_from<decltype(action_v)>;
          if (this->algorithm_step_ ==
              tmpl::index_of<actions_list, action>::value) {
            this->deadlock_analysis_next_iterable_action_ =
                pretty_type::name<action>();
          }
        });
      }
    };
    // Loop over all phases, once the current phase is found we perform the
    // algorithm in that phase until we are no longer able to because we are
    // waiting on data to be sent or because the algorithm has been marked as
    // terminated.
    EXPAND_PACK_LEFT_TO_RIGHT(invoke_for_phase(PhaseDepActionListsPack{}));
  } catch (const std::exception& exception) {
    initiate_shutdown(exception);
  }
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
std::string
DgElementArrayMember<Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
                     SimpleTagsFromOptions>::print_types() const {
  std::ostringstream os;
  os << "Algorithm type aliases:\n";
  os << "using all_actions_list = " << pretty_type::get_name<all_actions_list>()
     << ";\n";

  os << "using metavariables = " << pretty_type::get_name<metavariables>()
     << ";\n";
  os << "using inbox_tags_list = " << pretty_type::get_name<inbox_tags_list>()
     << ";\n";
  os << "using array_index = " << pretty_type::get_name<ElementId<Dim>>()
     << ";\n";
  os << "using parallel_component = "
     << pretty_type::get_name<ParallelComponent>() << ";\n";
  os << "using phase_dependent_action_lists = "
     << pretty_type::get_name<phase_dependent_action_lists>() << ";\n";
  os << "using all_cache_tags = " << pretty_type::get_name<all_cache_tags>()
     << ";\n";
  os << "using databox_type = " << pretty_type::get_name<databox_type>()
     << ";\n";
  return os.str();
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
std::string
DgElementArrayMember<Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
                     SimpleTagsFromOptions>::print_inbox() const {
  std::ostringstream os;
  os << "inboxes_ = " << inboxes_ << ";\n";
  return os.str();
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
std::string
DgElementArrayMember<Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
                     SimpleTagsFromOptions>::print_databox() const {
  std::ostringstream os;
  os << "box_:\n" << box_;
  return os.str();
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
template <typename PhaseDepActions, size_t... Is>
bool DgElementArrayMember<Dim, Metavariables,
                          tmpl::list<PhaseDepActionListsPack...>,
                          SimpleTagsFromOptions>::
    iterate_over_actions(const std::index_sequence<Is...> /*meta*/) {
  bool take_next_action = true;
  const auto helper = [this, &take_next_action](auto iteration) {
    constexpr size_t iter = decltype(iteration)::value;
    if (not(take_next_action and not this->terminate_ and
            not this->halt_algorithm_until_next_phase_ and
            this->algorithm_step_ == iter)) {
      return;
    }
    using actions_list = typename PhaseDepActions::action_list;
    using this_action = tmpl::at_c<actions_list, iter>;

    constexpr size_t phase_index =
        tmpl::index_of<phase_dependent_action_lists, PhaseDepActions>::value;
    this->performing_action_ = true;
    ++(this->algorithm_step_);
    // While the overhead from using the local entry method to enable
    // profiling is fairly small (<2%), we still avoid it when we aren't
    // tracing.
    // #ifdef SPECTRE_CHARM_PROJECTIONS
    //     if constexpr (Parallel::is_array<parallel_component>::value) {
    //       if (not this->thisProxy[array_index_]
    //                   .template invoke_iterable_action<
    //                       this_action, std::integral_constant<size_t,
    //                       phase_index>, std::integral_constant<size_t,
    //                       iter>>()) {
    //         take_next_action = false;
    //         --algorithm_step_;
    //       }
    //     } else {
    // #endif  // SPECTRE_CHARM_PROJECTIONS
    if (not invoke_iterable_action<this_action,
                                   std::integral_constant<size_t, phase_index>,
                                   std::integral_constant<size_t, iter>>()) {
      take_next_action = false;
      --(this->algorithm_step_);
    }
    // #ifdef SPECTRE_CHARM_PROJECTIONS
    //     }
    // #endif  // SPECTRE_CHARM_PROJECTIONS
    this->performing_action_ = false;
    // Wrap counter if necessary
    if (this->algorithm_step_ >= tmpl::size<actions_list>::value) {
      this->algorithm_step_ = 0;
    }
  };
  // In case of no Actions avoid compiler warning.
  (void)helper;
  // This is a template for loop for Is
  EXPAND_PACK_LEFT_TO_RIGHT(helper(std::integral_constant<size_t, Is>{}));
  return take_next_action;
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
template <typename ThisAction, typename PhaseIndex, typename DataBoxIndex>
bool DgElementArrayMember<Dim, Metavariables,
                          tmpl::list<PhaseDepActionListsPack...>,
                          SimpleTagsFromOptions>::invoke_iterable_action() {
  using phase_dep_action =
      tmpl::at_c<phase_dependent_action_lists, PhaseIndex::value>;
  using actions_list = typename phase_dep_action::action_list;

#ifdef SPECTRE_CHARM_PROJECTIONS
  if constexpr (Parallel::is_array<ParallelComponent>::value) {
    (void)Parallel::charmxx::RegisterInvokeIterableAction<
        ParallelComponent, ThisAction, PhaseIndex, DataBoxIndex>::registrar;
  }
#endif  // SPECTRE_CHARM_PROJECTIONS

  const auto& [requested_execution, next_action_step] = ThisAction::apply(
      box_, inboxes_, *Parallel::local_branch(global_cache_proxy_),
      std::as_const(this->element_id_), actions_list{},
      std::add_pointer_t<ParallelComponent>{});

  if (next_action_step.has_value()) {
    ASSERT(
        AlgorithmExecution::Retry != requested_execution,
        "Switching actions on Retry doesn't make sense. Specify std::nullopt "
        "as the second argument of the iterable action return type");
    this->algorithm_step_ = next_action_step.value();
  }

  switch (requested_execution) {
    case AlgorithmExecution::Continue:
      return true;
    case AlgorithmExecution::Retry:
      return false;
    case AlgorithmExecution::Pause: {
      auto& cache = *Parallel::local_branch(global_cache_proxy_);
      Parallel::local_synchronous_action<
          Parallel::Actions::SetTerminateOnElement>(
          Parallel::get_parallel_component<ParallelComponent>(cache),
          make_not_null(&cache), this->element_id_, true);
      return true;
    }
    case AlgorithmExecution::Halt: {
      // Need to make sure halt also gets propagated to the nodegroup
      this->halt_algorithm_until_next_phase_ = true;
      auto& cache = *Parallel::local_branch(global_cache_proxy_);
      Parallel::local_synchronous_action<
          Parallel::Actions::SetTerminateOnElement>(
          Parallel::get_parallel_component<ParallelComponent>(cache),
          make_not_null(&cache), this->element_id_, true);
      return true;
    }
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("No case for a Parallel::AlgorithmExecution with integral value "
            << static_cast<std::underlying_type_t<AlgorithmExecution>>(
                   requested_execution)
            << "\n");
      // LCOV_EXCL_STOP
  };
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
size_t DgElementArrayMember<
    Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
    SimpleTagsFromOptions>::number_of_actions_in_phase(const Parallel::Phase
                                                           phase) const {
  size_t number_of_actions = 0;
  const auto helper = [&number_of_actions, phase](auto pdal_v) {
    if (pdal_v.phase == phase) {
      number_of_actions = pdal_v.number_of_actions;
    }
  };
  EXPAND_PACK_LEFT_TO_RIGHT(helper(PhaseDepActionListsPack{}));
  return number_of_actions;
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
void DgElementArrayMember<
    Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
    SimpleTagsFromOptions>::initiate_shutdown(const std::exception& exception) {
  // In order to make it so that we can later run other actions for cleanup
  // (e.g. dumping data) we need to make sure that we enable running actions
  // again
  this->performing_action_ = false;
  // Send message to `Main` that we received an exception and set termination.
  auto* global_cache = Parallel::local_branch(global_cache_proxy_);
  if (UNLIKELY(global_cache == nullptr)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    CkError(
        "Global cache pointer is null. This is an internal inconsistency "
        "error. Please file an issue.");
    sys::abort("");
  }
  auto main_proxy = global_cache->get_main_proxy();
  if (UNLIKELY(not main_proxy.has_value())) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    CkError(
        "The main proxy has not been set in the global cache when terminating "
        "the component. This is an internal inconsistency error. Please file "
        "an issue.");
    sys::abort("");
  }
  const std::string message = MakeString{}
                              << "Message: " << exception.what() << "\nType: "
                              << pretty_type::get_runtime_type_name(exception);
  main_proxy.value().add_exception_message(message);
  auto& cache = *Parallel::local_branch(global_cache_proxy_);
  Parallel::local_synchronous_action<Parallel::Actions::SetTerminateOnElement>(
      Parallel::get_parallel_component<ParallelComponent>(cache),
      make_not_null(&cache), this->element_id_, true);
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
void DgElementArrayMember<Dim, Metavariables,
                          tmpl::list<PhaseDepActionListsPack...>,
                          SimpleTagsFromOptions>::pup(PUP::er& p) {
  DgElementArrayMemberBase<Dim>::pup(p);
  p | global_cache_proxy_;
  p | box_;
  p | inboxes_;
  if (p.isUnpacking()) {
    // Since we need the global cache to set the node, the derived class
    // does it instead of the base class.
    this->my_node_ =
        Parallel::my_node<size_t>(*Parallel::local_branch(global_cache_proxy_));
  }
}

template <size_t Dim, typename Metavariables,
          typename... PhaseDepActionListsPack, typename SimpleTagsFromOptions>
PUP::able::PUP_ID DgElementArrayMember<
    Dim, Metavariables, tmpl::list<PhaseDepActionListsPack...>,
    SimpleTagsFromOptions>::my_PUP_ID =  // NOLINT
    0;
/// \endcond
}  // namespace Parallel
