// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Protocols/ReductionDataFormatter.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver::async_solvers {
template <typename FieldsTag, typename OptionsGroup, typename SourceTag,
          typename Label, typename ArraySectionIdTag,
          bool ObserveInitialResidual>
struct CompleteStep;
}  // namespace LinearSolver::async_solvers
/// \endcond

/// Functionality shared between parallel linear solvers that have no global
/// synchronization points
namespace LinearSolver::async_solvers {

using reduction_data = Parallel::ReductionData<
    // Iteration
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Residual
    Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>;

template <typename OptionsGroup>
struct ResidualReductionFormatter
    : tt::ConformsTo<observers::protocols::ReductionDataFormatter> {
  using reduction_data = async_solvers::reduction_data;
  ResidualReductionFormatter() = default;
  ResidualReductionFormatter(std::string local_section_observation_key)
      : section_observation_key(std::move(local_section_observation_key)) {}
  std::string operator()(const size_t iteration_id,
                         const double residual) const {
    if (iteration_id == 0) {
      return Options::name<OptionsGroup>() + section_observation_key +
             " initialized with residual: " + get_output(residual);
    } else {
      return Options::name<OptionsGroup>() + section_observation_key + "(" +
             get_output(iteration_id) +
             ") iteration complete. Remaining residual: " +
             get_output(residual);
    }
  }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | section_observation_key; }
  std::string section_observation_key{};
};

template <typename OptionsGroup, typename ParallelComponent,
          typename Metavariables, typename ArrayIndex>
void contribute_to_residual_observation(
    const size_t iteration_id, const double residual_magnitude_square,
    Parallel::GlobalCache<Metavariables>& cache, const ArrayIndex& array_index,
    const std::string& section_observation_key) {
  auto& local_observer =
      *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
           cache)
           .ckLocalBranch();
  auto formatter =
      UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
               ::Verbosity::Quiet)
          ? std::make_optional(ResidualReductionFormatter<OptionsGroup>{
                section_observation_key})
          : std::nullopt;
  Parallel::simple_action<observers::Actions::ContributeReductionData>(
      local_observer,
      observers::ObservationId(
          iteration_id,
          pretty_type::get_name<OptionsGroup>() + section_observation_key),
      observers::ArrayComponentId{
          std::add_pointer_t<ParallelComponent>{nullptr},
          Parallel::ArrayIndex<ArrayIndex>(array_index)},
      std::string{"/" + Options::name<OptionsGroup>() +
                  section_observation_key + "Residuals"},
      std::vector<std::string>{"Iteration", "Residual"},
      reduction_data{iteration_id, residual_magnitude_square},
      std::move(formatter));
  if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
               ::Verbosity::Debug)) {
    if (iteration_id == 0) {
      Parallel::printf("%s %s initialized with local residual: %e\n",
                       get_output(array_index),
                       Options::name<OptionsGroup>() + section_observation_key,
                       sqrt(residual_magnitude_square));
    } else {
      Parallel::printf(
          "%s %s(%zu) iteration complete. Remaining local residual: %e\n",
          get_output(array_index),
          Options::name<OptionsGroup>() + section_observation_key, iteration_id,
          sqrt(residual_magnitude_square));
    }
  }
}

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using source_tag = SourceTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  using residual_magnitude_square_tag =
      LinearSolver::Tags::MagnitudeSquare<residual_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<Convergence::Tags::Iterations<OptionsGroup>>;

  using simple_tags = tmpl::list<Convergence::Tags::IterationId<OptionsGroup>,
                                 Convergence::Tags::HasConverged<OptionsGroup>,
                                 operator_applied_to_fields_tag>;
  using compute_tags =
      tmpl::list<LinearSolver::Tags::ResidualCompute<fields_tag, source_tag>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    // The `PrepareSolve` action populates these tags with initial
    // values, except for `operator_applied_to_fields_tag` which is
    // expected to be updated in every iteration of the algorithm
    Initialization::mutate_assign<
        tmpl::list<Convergence::Tags::IterationId<OptionsGroup>>>(
        make_not_null(&box), std::numeric_limits<size_t>::max());
    return std::make_tuple(std::move(box));
  }
};

template <typename OptionsGroup, typename ArraySectionIdTag = void>
struct RegisterObservers {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& box,
                const ArrayIndex& /*array_index*/) {
    // Get the observation key, or "Unused" if the element does not belong
    // to a section with this tag. In the latter case, no observations will
    // ever be contributed.
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    ASSERT(section_observation_key != "Unused",
           "The identifier 'Unused' is reserved to indicate that no "
           "observations with this key will be contributed. Use a different "
           "key, or change the identifier 'Unused' to something else.");
    return {
        observers::TypeOfObservation::Reduction,
        observers::ObservationKey{pretty_type::get_name<OptionsGroup>() +
                                  section_observation_key.value_or("Unused")}};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag,
          typename ArraySectionIdTag = void>
using RegisterElement = observers::Actions::RegisterWithObservers<
    RegisterObservers<OptionsGroup, ArraySectionIdTag>>;

/*!
 * \brief Prepare the asynchronous linear solver for a solve
 *
 * This action resets the asynchronous linear solver to its initial state, and
 * optionally observes the initial residual. If the initial residual should be
 * observed, both the `SourceTag` as well as the
 * `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, FieldsTag>`
 * must be up-to-date at the time this action is invoked.
 *
 * This action also provides an anchor point in the action list for looping the
 * linear solver, in the sense that the algorithm jumps back to the action
 * immediately following this one when a step is complete and the solver hasn't
 * yet converged (see `LinearSolver::async_solvers::CompleteStep`).
 *
 * \tparam FieldsTag The data `x` in the linear equation `Ax = b` to be solved.
 * Should hold the initial guess `x_0` at this point in the algorithm.
 * \tparam OptionsGroup An options group identifying the linear solver
 * \tparam SourceTag The data `b` in `Ax = b`
 * \tparam Label An optional compile-time label for the solver to distinguish
 * different solves with the same solver in the action list
 * \tparam ArraySectionIdTag Observe the residual norm separately for each
 * array section identified by this tag (see `Parallel::Section`). Set to `void`
 * to observe the residual norm over all elements of the array (default). The
 * section only affects observations of residuals and has no effect on the
 * solver algorithm.
 * \tparam ObserveInitialResidual Whether or not to observe the initial residual
 * `b - A x_0` (default: `true`). Disable when `b` or `A x_0` are not yet
 * available at the preparation stage.
 */
template <typename FieldsTag, typename OptionsGroup, typename SourceTag,
          typename Label = OptionsGroup, typename ArraySectionIdTag = void,
          bool ObserveInitialResidual = true>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>;

 public:
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    constexpr size_t iteration_id = 0;

    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s: Prepare solve\n", get_output(array_index),
                       Options::name<OptionsGroup>());
    }

    db::mutate<Convergence::Tags::IterationId<OptionsGroup>,
               Convergence::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> local_iteration_id,
           const gsl::not_null<Convergence::HasConverged*> has_converged,
           const size_t num_iterations) {
          *local_iteration_id = iteration_id;
          *has_converged =
              Convergence::HasConverged{num_iterations, iteration_id};
        },
        get<Convergence::Tags::Iterations<OptionsGroup>>(box));

    if constexpr (ObserveInitialResidual) {
      // Observe the initial residual even if no steps are going to be performed
      const std::optional<std::string> section_observation_key =
          observers::get_section_observation_key<ArraySectionIdTag>(box);
      if (section_observation_key.has_value()) {
        const auto& residual = get<residual_tag>(box);
        const double residual_magnitude_square =
            inner_product(residual, residual);
        contribute_to_residual_observation<OptionsGroup, ParallelComponent>(
            iteration_id, residual_magnitude_square, cache, array_index,
            *section_observation_key);
      }
    }

    // Skip steps entirely if the solve has already converged
    constexpr size_t step_end_index = tmpl::index_of<
        ActionList,
        CompleteStep<FieldsTag, OptionsGroup, SourceTag, Label,
                     ArraySectionIdTag, ObserveInitialResidual>>::value;
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, PrepareSolve>::value;
    return {std::move(box), false,
            get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                ? (step_end_index + 1)
                : (this_action_index + 1)};
  }
};

/*!
 * \brief Complete a step of the asynchronous linear solver
 *
 * This action prepares the next step of the asynchronous linear solver, and
 * observes the residual. To observe the correct residual, make sure the
 * `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, FieldsTag>` is
 * up-to-date at the time this action is invoked.
 *
 * This action checks if the algorithm has converged, i.e. it has completed the
 * requested number of steps. If it hasn't, the algorithm jumps back to the
 * action immediately following the `LinearSolver::async_solvers::PrepareSolve`
 * to perform another iteration. Make sure both actions use the same template
 * parameters.
 *
 * \tparam FieldsTag The data `x` in the linear equation `Ax = b` to be solved.
 * \tparam OptionsGroup An options group identifying the linear solver
 * \tparam SourceTag The data `b` in `Ax = b`
 * \tparam Label An optional compile-time label for the solver to distinguish
 * different solves with the same solver in the action list
 * \tparam ArraySectionIdTag Observe the residual norm separately for each
 * array section identified by this tag (see `Parallel::Section`). Set to `void`
 * to observe the residual norm over all elements of the array (default). The
 * section only affects observations of residuals and has no effect on the
 * solver algorithm.
 * \tparam ObserveInitialResidual Whether or not to observe the _initial_
 * residual `b - A x_0`. This parameter should match the one passed to
 * `PrepareSolve`.
 */
template <typename FieldsTag, typename OptionsGroup, typename SourceTag,
          typename Label = OptionsGroup, typename ArraySectionIdTag = void,
          bool ObserveInitialResidual = true>
struct CompleteStep {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Prepare for next iteration
    db::mutate<Convergence::Tags::IterationId<OptionsGroup>,
               Convergence::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id,
           const gsl::not_null<Convergence::HasConverged*> has_converged,
           const size_t num_iterations) {
          ++(*iteration_id);
          *has_converged =
              Convergence::HasConverged{num_iterations, *iteration_id};
        },
        get<Convergence::Tags::Iterations<OptionsGroup>>(box));

    // Observe element-local residual magnitude
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (section_observation_key.has_value()) {
      const size_t completed_iterations =
          get<Convergence::Tags::IterationId<OptionsGroup>>(box);
      const auto& residual = get<residual_tag>(box);
      const double residual_magnitude_square =
          inner_product(residual, residual);
      contribute_to_residual_observation<OptionsGroup, ParallelComponent>(
          completed_iterations, residual_magnitude_square, cache, array_index,
          *section_observation_key);
    }

    // Repeat steps until the solve has converged
    constexpr size_t step_begin_index =
        tmpl::index_of<
            ActionList,
            PrepareSolve<FieldsTag, OptionsGroup, SourceTag, Label,
                         ArraySectionIdTag, ObserveInitialResidual>>::value +
        1;
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, CompleteStep>::value;
    return {std::move(box), false,
            get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                ? (this_action_index + 1)
                : step_begin_index};
  }
};

}  // namespace LinearSolver::async_solvers
