// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>  // for declval

#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

/// \cond
namespace db {
template <typename TypeList>
class DataBox;
}  // namespace db
/// \endcond

namespace Parallel {
/// \cond
template <typename Metavariables>
class GlobalCache;
/// \endcond

/// The possible options for altering the current execution of the algorithm,
/// used in the return type of iterable actions.
enum class AlgorithmExecution {
  /// Leave the algorithm termination flag in its current state.
  Continue,
  /// Temporarily stop executing iterable actions, but try the same
  /// action again after receiving data from other distributed
  /// objects.  If the returned tuple has a third element, it must be
  /// std::numeric_limits<size_t>::max().
  Retry,
  /// Stop the execution of iterable actions, but allow entry methods
  /// (communication) to explicitly request restarting the execution.
  Pause,
  /// Stop the execution of iterable actions and do not allow their execution
  /// until after a phase change. Simple actions will still execute.
  Halt
};

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
 * - `FirstInputParameterType` the type of the first argument of the first
 * Action in the ActionsPack
 * - `AdditionalArgsList` the types of the arguments after the first
 * argument, which must all be the same for all Actions in the ActionsPack
 */
template <typename T>
struct build_action_return_types;

template <typename... AllActions>
struct build_action_return_types<tmpl::list<AllActions...>> {
  template <typename FirstInputParameterType, typename AdditionalArgsList>
  using f = typename Algorithm_detail::build_action_return_types_impl<
      sizeof...(AllActions) != 0, AdditionalArgsList>::
      template f<FirstInputParameterType, tmpl::list<>, AllActions...>;
};

template <typename PhaseType, PhaseType Phase, typename DataBoxTypes>
struct PhaseDependentDataBoxTypes {
  using databox_types = DataBoxTypes;
  using phase_type = PhaseType;
  static constexpr phase_type phase = Phase;
};

template <typename CumulativeDataboxTypes, typename PhaseDepActionLists,
          typename InputDataBox, typename InboxTagsList, typename Metavariables,
          typename ArrayIndex, typename ParallelComponent>
struct build_databox_types;

template <typename CumulativeDataboxTypes, typename InputDataBox,
          typename InboxTagsList, typename Metavariables, typename ArrayIndex,
          typename ParallelComponent>
struct build_databox_types<CumulativeDataboxTypes, tmpl::list<>, InputDataBox,
                           InboxTagsList, Metavariables, ArrayIndex,
                           ParallelComponent> {
  using type = CumulativeDataboxTypes;
};

// Loop over all the phase dependent action list structs and for each one
// compute the list of DataBox types. The loop is initiated by taking the last
// DataBox of the previous phase. The results for each phase are stored in the
// `PhaseDependentDataBoxTypes` struct.
//
// The DataBox type that leaves one phase is what enters the next phase in the
// order that the PDAL's (phase dependent action lists) are specified in the
// parallel component. Switching from one phase to any other is also supported
// as long as the DataBox types match up correctly. While the code could be
// generalized to generically support switching from any phase to any other
// phase with different DataBox types, this would be a large tensor product at
// compile time leading to significantly more complex code and also longer
// compile times.
template <typename CumulativeDataboxTypes, typename InputDataBox,
          typename InboxTagsList, typename Metavariables, typename ArrayIndex,
          typename ParallelComponent, typename CurrentPhaseDepActionList,
          typename... Rest>
struct build_databox_types<
    CumulativeDataboxTypes, tmpl::list<CurrentPhaseDepActionList, Rest...>,
    InputDataBox,

    InboxTagsList, Metavariables, ArrayIndex, ParallelComponent> {
  using additional_args_list =
      tmpl::list<tuples::tagged_tuple_from_typelist<InboxTagsList>,
                 Parallel::GlobalCache<Metavariables>, ArrayIndex,
                 typename CurrentPhaseDepActionList::action_list,
                 std::add_pointer_t<ParallelComponent>>;
  using action_list_of_this_phase =
      typename CurrentPhaseDepActionList::action_list;
  using databox_types_for_this_phase =
      typename build_action_return_types<action_list_of_this_phase>::template f<
          InputDataBox, additional_args_list>;
  using current_phase_dep_databox_types =
      PhaseDependentDataBoxTypes<typename CurrentPhaseDepActionList::phase_type,
                                 CurrentPhaseDepActionList::phase,
                                 databox_types_for_this_phase>;
  using cumulative_databox_types =
      tmpl::push_back<CumulativeDataboxTypes, current_phase_dep_databox_types>;

  using type = typename build_databox_types<
      cumulative_databox_types, tmpl::list<Rest...>,
      tmpl::back<databox_types_for_this_phase>, InboxTagsList, Metavariables,
      ArrayIndex, ParallelComponent>::type;
};

CREATE_IS_CALLABLE(is_ready)
CREATE_HAS_STATIC_MEMBER_VARIABLE(LoadBalancing)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(LoadBalancing)

// for checking the DataBox return of an iterable action
template <typename FirstIterableActionType>
struct is_databox_or_databox_rvalue : std::false_type {};

template <typename TypeList>
struct is_databox_or_databox_rvalue<db::DataBox<TypeList>>
    : std::true_type {};

template <typename TypeList>
struct is_databox_or_databox_rvalue<db::DataBox<TypeList>&&>
    : std::true_type {};

// for checking that iterable action has the correct return type. The second
// template parameter is unused, but required to be passed so that the generated
// template error from a failing static_assert displays the action for which the
// return type is invalid.
template <typename ParallelComponentType, typename ActionType,
          typename GeneralType>
struct check_iterable_action_return_type : std::false_type {};

template <typename ParallelComponentType, typename ActionType,
          typename FirstType>
struct check_iterable_action_return_type<ParallelComponentType, ActionType,
                                         std::tuple<FirstType>>
    : is_databox_or_databox_rvalue<FirstType> {};

template <typename ParallelComponentType, typename ActionType,
          typename FirstType>
struct check_iterable_action_return_type<ParallelComponentType, ActionType,
                                         std::tuple<FirstType, bool>>
    : is_databox_or_databox_rvalue<FirstType> {};

template <typename ParallelComponentType, typename ActionType,
          typename FirstType>
struct check_iterable_action_return_type<
    ParallelComponentType, ActionType,
    std::tuple<FirstType, AlgorithmExecution>>
    : is_databox_or_databox_rvalue<FirstType> {};

template <typename ParallelComponentType, typename ActionType,
          typename FirstType>
struct check_iterable_action_return_type<ParallelComponentType, ActionType,
                                         std::tuple<FirstType, bool, size_t>>
    : is_databox_or_databox_rvalue<FirstType> {};

template <typename ParallelComponentType, typename ActionType,
          typename FirstType>
struct check_iterable_action_return_type<
    ParallelComponentType, ActionType,
    std::tuple<FirstType, AlgorithmExecution, size_t>>
    : is_databox_or_databox_rvalue<FirstType> {};
}  // namespace Algorithm_detail
}  // namespace Parallel
