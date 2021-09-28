// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <variant>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GetSection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/ResidualMonitorActions.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/Tags/InboxTags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace NonlinearSolver::newton_raphson::detail {
template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct ResidualMonitor;
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct ReceiveInitialHasConverged;
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct PrepareStep;
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct Globalize;
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct CompleteStep;
}  // namespace NonlinearSolver::newton_raphson::detail
/// \endcond

namespace NonlinearSolver::newton_raphson::detail {

using ResidualReductionData = Parallel::ReductionData<
    // Iteration ID
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Globalization iteration ID
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Residual magnitude square
    Parallel::ReductionDatum<double, funcl::Plus<>>,
    // Step length
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>>;

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using source_tag = SourceTag;
  using nonlinear_operator_applied_to_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using correction_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;
  using globalization_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Globalization, fields_tag>;

 public:
  using simple_tags =
      tmpl::list<Convergence::Tags::IterationId<OptionsGroup>,
                 Convergence::Tags::HasConverged<OptionsGroup>,
                 nonlinear_operator_applied_to_fields_tag, correction_tag,
                 NonlinearSolver::Tags::Globalization<
                     Convergence::Tags::IterationId<OptionsGroup>>,
                 NonlinearSolver::Tags::StepLength<OptionsGroup>,
                 globalization_fields_tag>;
  using compute_tags = tmpl::list<
      NonlinearSolver::Tags::ResidualCompute<fields_tag, source_tag>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    ::Initialization::mutate_assign<
        tmpl::list<Convergence::Tags::IterationId<OptionsGroup>,
                   NonlinearSolver::Tags::Globalization<
                       Convergence::Tags::IterationId<OptionsGroup>>,
                   NonlinearSolver::Tags::StepLength<OptionsGroup>>>(
        make_not_null(&box), std::numeric_limits<size_t>::max(),
        std::numeric_limits<size_t>::max(),
        std::numeric_limits<double>::signaling_NaN());
    return std::make_tuple(std::move(box));
  }
};

// Reset to the initial state of the algorithm.
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;
  using nonlinear_residual_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution,
                    size_t>
  apply(db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& array_index, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    db::mutate<Convergence::Tags::IterationId<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id) { *iteration_id = 0; });

    // Skip the initial reduction on elements that are not part of the section
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::Section<ParallelComponent,
                                              ArraySectionIdTag>>(box)
                  .has_value()) {
        constexpr size_t receive_initial_index = tmpl::index_of<
            ActionList,
            ReceiveInitialHasConverged<FieldsTag, OptionsGroup, Label,
                                       ArraySectionIdTag>>::value;
        return {std::move(box), Parallel::AlgorithmExecution::Continue,
                receive_initial_index};
      }
    }

    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s %s: Prepare solve\n", get_output(array_index),
                       Options::name<OptionsGroup>());
    }

    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, PrepareSolve>::value;
    return {std::move(box), Parallel::AlgorithmExecution::Continue,
            this_action_index + 1};
  }
};

// To determine the initial residual magnitude and to check if the algorithm has
// already converged, we perform a global reduction to the `ResidualMonitor`.
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct SendInitialResidualMagnitude {
 private:
  using fields_tag = FieldsTag;
  using nonlinear_residual_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Perform a global reduction to compute the initial residual magnitude
    const auto& residual = db::get<nonlinear_residual_tag>(box);
    const double local_residual_magnitude_square =
        LinearSolver::inner_product(residual, residual);
    ResidualReductionData reduction_data{0, 0, local_residual_magnitude_square,
                                         1.};
    auto& section = Parallel::get_section<ParallelComponent, ArraySectionIdTag>(
        make_not_null(&box));
    Parallel::contribute_to_reduction<
        CheckResidualMagnitude<FieldsTag, OptionsGroup, ParallelComponent>>(
        std::move(reduction_data),
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache),
        make_not_null(&section));
    return {std::move(box)};
  }
};

// Wait for the broadcast from the `ResidualMonitor` to complete the preparation
// for the solve. We skip the solve altogether if the algorithm has already
// converged.
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct ReceiveInitialHasConverged {
  using inbox_tags = tmpl::list<Tags::GlobalizationResult<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution,
                    size_t>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    auto& inbox = get<Tags::GlobalizationResult<OptionsGroup>>(inboxes);
    if (inbox.find(db::get<Convergence::Tags::IterationId<OptionsGroup>>(
            box)) == inbox.end()) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry,
              std::numeric_limits<size_t>::max()};
    }

    // Retrieve reduction data from inbox
    auto globalization_result = std::move(inbox.extract(iteration_id).mapped());
    ASSERT(
        std::holds_alternative<Convergence::HasConverged>(globalization_result),
        "No globalization should occur for the initial residual. This is a "
        "bug, so please file an issue.");
    auto& has_converged = get<Convergence::HasConverged>(globalization_result);
    db::mutate<Convergence::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [&has_converged](const gsl::not_null<Convergence::HasConverged*>
                             local_has_converged) {
          *local_has_converged = std::move(has_converged);
        });

    // Skip steps entirely if the solve has already converged
    constexpr size_t complete_step_index =
        tmpl::index_of<ActionList, CompleteStep<FieldsTag, OptionsGroup, Label,
                                                ArraySectionIdTag>>::value;
    if (get<Convergence::Tags::HasConverged<OptionsGroup>>(box)) {
      return {std::move(box), Parallel::AlgorithmExecution::Continue,
              complete_step_index + 1};
    }

    // Skip the solve entirely on elements that are not part of the section. To
    // do so, we skip ahead to the `SolveLinearization` action list between
    // `PrepareStep` and `PerformStep`. Those actions need a chance to run even
    // on elements that are not part of the section, because they may take part
    // in preconditioning (see Multigrid preconditioner).
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::Section<ParallelComponent,
                                              ArraySectionIdTag>>(box)
                  .has_value()) {
        db::mutate<Convergence::Tags::IterationId<OptionsGroup>>(
            make_not_null(&box),
            [](const gsl::not_null<size_t*> local_iteration_id) {
              *local_iteration_id = 1;
            });
        constexpr size_t prepare_step_index =
            tmpl::index_of<ActionList,
                           PrepareStep<FieldsTag, OptionsGroup, Label,
                                       ArraySectionIdTag>>::value;
        return {std::move(box), Parallel::AlgorithmExecution::Continue,
                prepare_step_index + 1};
      }
    }

    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, ReceiveInitialHasConverged>::value;
    return {std::move(box), Parallel::AlgorithmExecution::Continue,
            this_action_index + 1};
  }
};

// Prepare the next Newton-Raphson step. In particular, we prepare the DataBox
// for the linear solver which will run after this action to solve the
// linearized problem for the `correction_tag`. The source for the linear
// solver is the residual, which is in the DataBox as a compute tag so needs no
// preparation. We only need to set the initial guess.
//
// We also prepare the line-search globalization here. Since we don't know if
// a step will be sufficient before taking it, we have to store the original
// field values.
//
// The algorithm jumps back to this action from `CompleteStep` to continue
// iterating the nonlinear solve.
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct PrepareStep {
 private:
  using fields_tag = FieldsTag;
  using correction_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;
  using linear_operator_applied_to_correction_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, correction_tag>;
  using globalization_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Globalization, fields_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<NonlinearSolver::Tags::DampingFactor<OptionsGroup>,
                 logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s %s(%zu): Prepare step\n", get_output(array_index),
          Options::name<OptionsGroup>(),
          db::get<Convergence::Tags::IterationId<OptionsGroup>>(box) + 1);
    }

    db::mutate<Convergence::Tags::IterationId<OptionsGroup>, correction_tag,
               linear_operator_applied_to_correction_tag,
               NonlinearSolver::Tags::Globalization<
                   Convergence::Tags::IterationId<OptionsGroup>>,
               NonlinearSolver::Tags::StepLength<OptionsGroup>,
               globalization_fields_tag>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id, const auto correction,
           const auto linear_operator_applied_to_correction,
           const gsl::not_null<size_t*> globalization_iteration_id,
           const gsl::not_null<double*> step_length,
           const auto globalization_fields, const auto& fields,
           const double damping_factor) {
          ++(*iteration_id);
          // Begin the linear solve with a zero initial guess
          *correction =
              make_with_value<typename correction_tag::type>(fields, 0.);
          // Since the initial guess is zero, we don't need to apply the linear
          // operator to it but can just set it to zero as well. Linear things
          // are nice :)
          *linear_operator_applied_to_correction = make_with_value<
              typename linear_operator_applied_to_correction_tag::type>(fields,
                                                                        0.);
          // Prepare line search globalization
          *globalization_iteration_id = 0;
          *step_length = damping_factor;
          *globalization_fields = fields;
        },
        db::get<fields_tag>(box),
        db::get<NonlinearSolver::Tags::DampingFactor<OptionsGroup>>(box));
    return {std::move(box)};
  }
};

// Between `PrepareStep` and this action the linear solver has run, so the
// `correction_tag` is now filled with the solution to the linearized problem.
// We just take a step in the direction of the correction.
//
// The `Globalize` action will jump back here in case the step turned out to not
// decrease the residual sufficiently. It will have adjusted the step length, so
// we can just try again with a step of that length.
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct PerformStep {
 private:
  using fields_tag = FieldsTag;
  using correction_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;
  using globalization_fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Globalization, fields_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution,
                    size_t>
  apply(db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& array_index, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    // Skip to the end of the step on elements that are not part of the section
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<
              Parallel::Tags::Section<ParallelComponent, ArraySectionIdTag>>(
              box)) {
        constexpr size_t globalize_index =
            tmpl::index_of<ActionList, Globalize<FieldsTag, OptionsGroup, Label,
                                                 ArraySectionIdTag>>::value;
        return {std::move(box), Parallel::AlgorithmExecution::Continue,
                globalize_index};
      }
    }

    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s %s(%zu): Perform step with length: %g\n", get_output(array_index),
          Options::name<OptionsGroup>(),
          db::get<Convergence::Tags::IterationId<OptionsGroup>>(box),
          db::get<NonlinearSolver::Tags::StepLength<OptionsGroup>>(box));
    }

    // Apply the correction that the linear solve has determined to attempt
    // improving the nonlinear solution
    db::mutate<fields_tag>(
        make_not_null(&box),
        [](const auto fields, const auto& correction, const double step_length,
           const auto& globalization_fields) {
          *fields = globalization_fields + step_length * correction;
        },
        db::get<correction_tag>(box),
        db::get<NonlinearSolver::Tags::StepLength<OptionsGroup>>(box),
        db::get<globalization_fields_tag>(box));

    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, PerformStep>::value;
    return {std::move(box), Parallel::AlgorithmExecution::Continue,
            this_action_index + 1};
  }
};

// Between `PerformStep` and this action the nonlinear operator has been applied
// to the updated fields. The residual is up-to-date because it is a compute
// tag, so at this point we need to check if the step has sufficiently reduced
// the residual. We perform a global reduction to the `ResidualMonitor` for this
// purpose.
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct ContributeToResidualMagnitudeReduction {
 private:
  using fields_tag = FieldsTag;
  using nonlinear_residual_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& residual = db::get<nonlinear_residual_tag>(box);
    const double local_residual_magnitude_square =
        LinearSolver::inner_product(residual, residual);
    ResidualReductionData reduction_data{
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box),
        db::get<NonlinearSolver::Tags::Globalization<
            Convergence::Tags::IterationId<OptionsGroup>>>(box),
        local_residual_magnitude_square,
        db::get<NonlinearSolver::Tags::StepLength<OptionsGroup>>(box)};
    auto& section = Parallel::get_section<ParallelComponent, ArraySectionIdTag>(
        make_not_null(&box));
    Parallel::contribute_to_reduction<
        CheckResidualMagnitude<FieldsTag, OptionsGroup, ParallelComponent>>(
        std::move(reduction_data),
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache),
        make_not_null(&section));
    return {std::move(box)};
  }
};

// Wait for the `ResidualMonitor` to broadcast whether or not it has determined
// the step has sufficiently decreased the residual. If it has, we just proceed
// to `CompleteStep`. If it hasn't, the `ResidualMonitor` has also sent the new
// step length along, so we mutate it in the DataBox and jump back to
// `PerformStep` to try again with the updated step length.
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct Globalize {
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;
  using inbox_tags = tmpl::list<Tags::GlobalizationResult<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution,
                    size_t>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& array_index, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    auto& inbox = get<Tags::GlobalizationResult<OptionsGroup>>(inboxes);
    const size_t iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (inbox.find(iteration_id) == inbox.end()) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry,
              std::numeric_limits<size_t>::max()};
    }

    // Retrieve reduction data from inbox
    auto globalization_result = std::move(inbox.extract(iteration_id).mapped());

    // Elements that are not part of the section jump directly to the
    // `SolveLinearization` for the next step.
    constexpr size_t complete_step_index =
        tmpl::index_of<ActionList, CompleteStep<FieldsTag, OptionsGroup, Label,
                                                ArraySectionIdTag>>::value;
    constexpr size_t prepare_step_index =
        tmpl::index_of<ActionList, PrepareStep<FieldsTag, OptionsGroup, Label,
                                               ArraySectionIdTag>>::value;
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<
              Parallel::Tags::Section<ParallelComponent, ArraySectionIdTag>>(
              box)) {
        const bool globalization_is_complete =
            std::holds_alternative<Convergence::HasConverged>(
                globalization_result);
        // Wait until globalization is complete
        if (not globalization_is_complete) {
          return {std::move(box), Parallel::AlgorithmExecution::Retry,
                  std::numeric_limits<size_t>::max()};
        }
        auto& has_converged =
            get<Convergence::HasConverged>(globalization_result);

        db::mutate<Convergence::Tags::HasConverged<OptionsGroup>,
                   Convergence::Tags::IterationId<OptionsGroup>>(
            make_not_null(&box),
            [&has_converged](const gsl::not_null<Convergence::HasConverged*>
                                 local_has_converged,
                             const gsl::not_null<size_t*> local_iteration_id) {
              *local_has_converged = std::move(has_converged);
              ++(*local_iteration_id);
            });

        return {std::move(box), Parallel::AlgorithmExecution::Continue,
                get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                    ? (complete_step_index + 1)
                    : (prepare_step_index + 1)};
      }
    }

    if (std::holds_alternative<double>(globalization_result)) {
      if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                   ::Verbosity::Debug)) {
        Parallel::printf(
            "%s %s(%zu): Globalize(%zu)\n", get_output(array_index),
            Options::name<OptionsGroup>(),
            db::get<Convergence::Tags::IterationId<OptionsGroup>>(box),
            db::get<NonlinearSolver::Tags::Globalization<
                Convergence::Tags::IterationId<OptionsGroup>>>(box));
      }

      // Update the step length
      db::mutate<NonlinearSolver::Tags::StepLength<OptionsGroup>,
                 NonlinearSolver::Tags::Globalization<
                     Convergence::Tags::IterationId<OptionsGroup>>>(
          make_not_null(&box),
          [&globalization_result](
              const gsl::not_null<double*> step_length,
              const gsl::not_null<size_t*> globalization_iteration_id) {
            *step_length = get<double>(globalization_result);
            ++(*globalization_iteration_id);
          });
      // Continue globalization by taking a step with the updated step length
      // and checking the residual again
      constexpr size_t perform_step_index =
          tmpl::index_of<ActionList, PerformStep<FieldsTag, OptionsGroup, Label,
                                                 ArraySectionIdTag>>::value;
      return {std::move(box), Parallel::AlgorithmExecution::Continue,
              perform_step_index};
    }

    // At this point globalization is complete, so we proceed with the algorithm
    auto& has_converged = get<Convergence::HasConverged>(globalization_result);

    db::mutate<Convergence::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [&has_converged](const gsl::not_null<Convergence::HasConverged*>
                             local_has_converged) {
          *local_has_converged = std::move(has_converged);
        });

    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, Globalize>::value;
    return {std::move(box), Parallel::AlgorithmExecution::Continue,
            this_action_index + 1};
  }
};

// Jump back to `PrepareStep` to continue iterating if the algorithm has not yet
// converged, or complete the solve and proceed with the action list if it has
// converged. This is a separate action because the user has the opportunity to
// insert actions before the step completes, for example to do observations.
template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename ArraySectionIdTag>
struct CompleteStep {
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s %s(%zu): Complete step\n", get_output(array_index),
          Options::name<OptionsGroup>(),
          db::get<Convergence::Tags::IterationId<OptionsGroup>>(box));
    }

    // Repeat steps until the solve has converged
    constexpr size_t prepare_step_index =
        tmpl::index_of<ActionList, PrepareStep<FieldsTag, OptionsGroup, Label,
                                               ArraySectionIdTag>>::value;
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, CompleteStep>::value;
    return {std::move(box), false,
            get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                ? (this_action_index + 1)
                : prepare_step_index};
  }
};

}  // namespace NonlinearSolver::newton_raphson::detail
