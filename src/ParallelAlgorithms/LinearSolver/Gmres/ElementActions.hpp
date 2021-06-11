// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/ResidualMonitorActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Tags/InboxTags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver::gmres::detail {
template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct ResidualMonitor;
template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label>
struct NormalizeOperandAndUpdateField;
}  // namespace LinearSolver::gmres::detail
/// \endcond

namespace LinearSolver::gmres::detail {

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label, typename SourceTag>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;
  using initial_fields_tag = db::add_tag_prefix<::Tags::Initial, fields_tag>;
  using source_tag = SourceTag;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<Convergence::Tags::IterationId<OptionsGroup>, operand_tag,
               initial_fields_tag, basis_history_tag>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id, const auto operand,
           const auto initial_fields, const auto basis_history,
           const auto& source, const auto& operator_applied_to_fields,
           const auto& fields) noexcept {
          *iteration_id = 0;
          *operand = source - operator_applied_to_fields;
          *initial_fields = fields;
          *basis_history = typename basis_history_tag::type{};
        },
        get<source_tag>(box), get<operator_applied_to_fields_tag>(box),
        get<fields_tag>(box));

    Parallel::contribute_to_reduction<InitializeResidualMagnitude<
        FieldsTag, OptionsGroup, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>{
            inner_product(get<operand_tag>(box), get<operand_tag>(box))},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache));

    if constexpr (Preconditioned) {
      using preconditioned_operand_tag =
          db::add_tag_prefix<LinearSolver::Tags::Preconditioned, operand_tag>;
      using preconditioned_basis_history_tag =
          LinearSolver::Tags::KrylovSubspaceBasis<preconditioned_operand_tag>;

      db::mutate<preconditioned_basis_history_tag>(
          make_not_null(&box),
          [](const auto preconditioned_basis_history) noexcept {
            *preconditioned_basis_history =
                typename preconditioned_basis_history_tag::type{};
          });
    }

    return {std::move(box)};
  }
};

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label>
struct NormalizeInitialOperand {
 private:
  using fields_tag = FieldsTag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;

 public:
  using inbox_tags = tmpl::list<Tags::InitialOrthogonalization<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution,
                    size_t>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
    auto& inbox = get<Tags::InitialOrthogonalization<OptionsGroup>>(inboxes);
    if (inbox.find(db::get<Convergence::Tags::IterationId<OptionsGroup>>(
            box)) == inbox.end()) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry,
              std::numeric_limits<size_t>::max()};
    }

    auto received_data = std::move(
        inbox
            .extract(db::get<Convergence::Tags::IterationId<OptionsGroup>>(box))
            .mapped());
    const double residual_magnitude = get<0>(received_data);
    auto& has_converged = get<1>(received_data);

    db::mutate<operand_tag, basis_history_tag,
               Convergence::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box), [residual_magnitude, &has_converged](
                                 const auto operand, const auto basis_history,
                                 const gsl::not_null<Convergence::HasConverged*>
                                     local_has_converged) noexcept {
          *operand /= residual_magnitude;
          basis_history->push_back(*operand);
          *local_has_converged = std::move(has_converged);
        });

    // Skip steps entirely if the solve has already converged
    constexpr size_t step_end_index = tmpl::index_of<
        ActionList, NormalizeOperandAndUpdateField<
                        FieldsTag, OptionsGroup, Preconditioned, Label>>::value;
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, NormalizeInitialOperand>::value;
    return {std::move(box), Parallel::AlgorithmExecution::Continue,
            get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                ? (step_end_index + 1)
                : (this_action_index + 1)};
  }
};

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label>
struct PrepareStep {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if constexpr (Preconditioned) {
      using fields_tag = FieldsTag;
      using operand_tag =
          db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
      using preconditioned_operand_tag =
          db::add_tag_prefix<LinearSolver::Tags::Preconditioned, operand_tag>;
      using operator_tag = db::add_tag_prefix<
          LinearSolver::Tags::OperatorAppliedTo,
          std::conditional_t<Preconditioned, preconditioned_operand_tag,
                             operand_tag>>;

      db::mutate<preconditioned_operand_tag, operator_tag>(
          make_not_null(&box),
          [](const auto preconditioned_operand,
             const auto operator_applied_to_operand,
             const auto& operand) noexcept {
            // Start the preconditioner at zero because we have no reason to
            // expect the remaining residual to have a particular form.
            // Another possibility would be to start the preconditioner with an
            // initial guess equal to its source, so not running the
            // preconditioner at all means it is the identity, but that approach
            // appears to yield worse results.
            *preconditioned_operand =
                make_with_value<typename preconditioned_operand_tag::type>(
                    operand, 0.);
            // Also set the operator applied to the initial preconditioned
            // operand to zero because it's linear. This may save the
            // preconditioner an operator application if it's optimized for
            // this.
            *operator_applied_to_operand =
                make_with_value<typename operator_tag::type>(operand, 0.);
          },
          get<operand_tag>(box));
    }
    return {std::move(box)};
  }
};

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label>
struct PerformStep {
 private:
  using fields_tag = FieldsTag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using preconditioned_operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Preconditioned, operand_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using operator_tag = db::add_tag_prefix<
        LinearSolver::Tags::OperatorAppliedTo,
        std::conditional_t<Preconditioned, preconditioned_operand_tag,
                           operand_tag>>;
    using orthogonalization_iteration_id_tag =
        LinearSolver::Tags::Orthogonalization<
            Convergence::Tags::IterationId<OptionsGroup>>;
    using basis_history_tag =
        LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;

    if constexpr (Preconditioned) {
      using preconditioned_basis_history_tag =
          LinearSolver::Tags::KrylovSubspaceBasis<preconditioned_operand_tag>;

      db::mutate<preconditioned_basis_history_tag>(
          make_not_null(&box),
          [](const auto preconditioned_basis_history,
             const auto& preconditioned_operand) noexcept {
            preconditioned_basis_history->push_back(preconditioned_operand);
          },
          get<preconditioned_operand_tag>(box));
    }

    db::mutate<operand_tag, orthogonalization_iteration_id_tag>(
        make_not_null(&box),
        [](const auto operand,
           const gsl::not_null<size_t*> orthogonalization_iteration_id,
           const auto& operator_action) noexcept {
          *operand = typename operand_tag::type(operator_action);
          *orthogonalization_iteration_id = 0;
        },
        get<operator_tag>(box));

    Parallel::contribute_to_reduction<
        StoreOrthogonalization<FieldsTag, OptionsGroup, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            get<Convergence::Tags::IterationId<OptionsGroup>>(box),
            get<orthogonalization_iteration_id_tag>(box),
            inner_product(get<basis_history_tag>(box)[0],
                          get<operand_tag>(box))},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache));

    return {std::move(box)};
  }
};

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label>
struct OrthogonalizeOperand {
 private:
  using fields_tag = FieldsTag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using orthogonalization_iteration_id_tag =
      LinearSolver::Tags::Orthogonalization<
          Convergence::Tags::IterationId<OptionsGroup>>;
  using basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;

 public:
  using inbox_tags = tmpl::list<Tags::Orthogonalization<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution,
                    size_t>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        Parallel::GlobalCache<Metavariables>& cache,
        const ArrayIndex& array_index, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
    auto& inbox = get<Tags::Orthogonalization<OptionsGroup>>(inboxes);
    if (inbox.find(db::get<Convergence::Tags::IterationId<OptionsGroup>>(
            box)) == inbox.end()) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry,
              std::numeric_limits<size_t>::max()};
    }

    const double orthogonalization = std::move(
        inbox
            .extract(db::get<Convergence::Tags::IterationId<OptionsGroup>>(box))
            .mapped());

    db::mutate<operand_tag, orthogonalization_iteration_id_tag>(
        make_not_null(&box),
        [orthogonalization](
            const auto operand,
            const gsl::not_null<size_t*> orthogonalization_iteration_id,
            const auto& basis_history) noexcept {
          *operand -= orthogonalization *
                      gsl::at(basis_history, *orthogonalization_iteration_id);
          ++(*orthogonalization_iteration_id);
        },
        get<basis_history_tag>(box));

    const auto& next_orthogonalization_iteration_id =
        get<orthogonalization_iteration_id_tag>(box);
    const auto& iteration_id =
        get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    const bool orthogonalization_complete =
        next_orthogonalization_iteration_id == iteration_id + 1;
    const double local_orthogonalization =
        inner_product(orthogonalization_complete
                          ? get<operand_tag>(box)
                          : gsl::at(get<basis_history_tag>(box),
                                    next_orthogonalization_iteration_id),
                      get<operand_tag>(box));

    Parallel::contribute_to_reduction<
        StoreOrthogonalization<FieldsTag, OptionsGroup, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            iteration_id, next_orthogonalization_iteration_id,
            local_orthogonalization},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache));

    // Repeat this action until orthogonalization is complete
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, OrthogonalizeOperand>::value;
    return {std::move(box), Parallel::AlgorithmExecution::Continue,
            orthogonalization_complete ? (this_action_index + 1)
                                       : this_action_index};
  }
};

template <typename FieldsTag, typename OptionsGroup, bool Preconditioned,
          typename Label>
struct NormalizeOperandAndUpdateField {
 private:
  using fields_tag = FieldsTag;
  using initial_fields_tag = db::add_tag_prefix<::Tags::Initial, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using preconditioned_operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Preconditioned, operand_tag>;
  using basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<operand_tag>;
  using preconditioned_basis_history_tag =
      LinearSolver::Tags::KrylovSubspaceBasis<std::conditional_t<
          Preconditioned, preconditioned_operand_tag, operand_tag>>;

 public:
  using inbox_tags = tmpl::list<Tags::FinalOrthogonalization<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution,
                    size_t>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
    auto& inbox = get<Tags::FinalOrthogonalization<OptionsGroup>>(inboxes);
    if (inbox.find(db::get<Convergence::Tags::IterationId<OptionsGroup>>(
            box)) == inbox.end()) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry,
              std::numeric_limits<size_t>::max()};
    }

    // Retrieve reduction data from inbox
    auto received_data = std::move(
        inbox
            .extract(db::get<Convergence::Tags::IterationId<OptionsGroup>>(box))
            .mapped());
    const double normalization = get<0>(received_data);
    const auto& minres = get<1>(received_data);
    auto& has_converged = get<2>(received_data);

    db::mutate<operand_tag, basis_history_tag, fields_tag,
               Convergence::Tags::IterationId<OptionsGroup>,
               Convergence::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [normalization, &minres, &has_converged](
            const auto operand, const auto basis_history, const auto field,
            const gsl::not_null<size_t*> iteration_id,
            const gsl::not_null<Convergence::HasConverged*> local_has_converged,
            const auto& initial_field,
            const auto& preconditioned_basis_history) noexcept {
          // Avoid an FPE if the new operand norm is exactly zero. In that case
          // the problem is solved and the algorithm will terminate (see
          // Proposition 9.3 in \cite Saad2003). Since there will be no next
          // iteration we don't need to normalize the operand.
          if (LIKELY(normalization > 0.)) {
            *operand /= normalization;
          }
          basis_history->push_back(*operand);
          *field = initial_field;
          for (size_t i = 0; i < minres.size(); i++) {
            *field += minres[i] * gsl::at(preconditioned_basis_history, i);
          }
          ++(*iteration_id);
          *local_has_converged = std::move(has_converged);
        },
        get<initial_fields_tag>(box),
        get<preconditioned_basis_history_tag>(box));

    // Repeat steps until the solve has converged
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, NormalizeOperandAndUpdateField>::value;
    constexpr size_t prepare_step_index =
        tmpl::index_of<ActionList, PrepareStep<FieldsTag, OptionsGroup,
                                               Preconditioned, Label>>::value;
    return {std::move(box), Parallel::AlgorithmExecution::Continue,
            get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                ? (this_action_index + 1)
                : prepare_step_index};
  }
};

}  // namespace LinearSolver::gmres::detail
