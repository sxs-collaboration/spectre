// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DenseVector.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres/ResidualMonitorActions.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace gmres_detail {
template <typename Metavariables>
struct ResidualMonitor;
}  // namespace gmres_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace gmres_detail {

struct NormalizeInitialOperand {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl2::flat_any_v<cpp17::is_same_v<
                db::add_tag_prefix<LinearSolver::Tags::Operand,
                                   typename Metavariables::system::fields_tag>,
                DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double residual_magnitude) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
    using basis_history_tag =
        LinearSolver::Tags::KrylovSubspaceBasis<fields_tag>;
    using residual_magnitude_tag = db::add_tag_prefix<
        LinearSolver::Tags::Magnitude,
        db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;

    db::mutate<operand_tag, basis_history_tag, residual_magnitude_tag>(
        make_not_null(&box),
        [residual_magnitude](
            const gsl::not_null<db::item_type<operand_tag>*> operand,
            const gsl::not_null<db::item_type<basis_history_tag>*>
                basis_history,
            const gsl::not_null<double*> local_residual_magnitude) noexcept {
          *operand /= residual_magnitude;
          basis_history->push_back(*operand);
          *local_residual_magnitude = residual_magnitude;
        });
  }
};

struct PerformStep {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
    using operator_tag =
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;
    using basis_history_tag =
        LinearSolver::Tags::KrylovSubspaceBasis<fields_tag>;

    db::mutate<operand_tag>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<operand_tag>*> operand,
           const db::item_type<operator_tag>& operator_action) noexcept {
          *operand = db::item_type<operand_tag>(operator_action);
        },
        get<operator_tag>(box));

    Parallel::contribute_to_reduction<
        StoreOrthogonalization<ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{inner_product(
            get<basis_history_tag>(box)[0], get<operand_tag>(box))},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));

    return std::tuple<db::DataBox<DbTagsList>&&, bool>(std::move(box), true);
  }
};

struct OrthogonalizeOperand {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl2::flat_any_v<cpp17::is_same_v<
                db::add_tag_prefix<LinearSolver::Tags::Operand,
                                   typename Metavariables::system::fields_tag>,
                DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double orthogonalization) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
    using orthogonalization_iteration_id_tag =
        db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                           LinearSolver::Tags::IterationId>;
    using basis_history_tag =
        LinearSolver::Tags::KrylovSubspaceBasis<fields_tag>;

    db::mutate<operand_tag, orthogonalization_iteration_id_tag>(
        make_not_null(&box),
        [orthogonalization](
            const gsl::not_null<db::item_type<operand_tag>*> operand,
            const gsl::not_null<IterationId*> orthogonalization_iteration_id,
            const db::item_type<basis_history_tag>& basis_history) noexcept {
          *operand -= orthogonalization *
                      gsl::at(basis_history,
                              orthogonalization_iteration_id->step_number);
          orthogonalization_iteration_id->step_number++;
        },
        get<basis_history_tag>(box));

    const auto next_orthogonalization_iteration_id =
        get<orthogonalization_iteration_id_tag>(box).step_number;
    const auto iteration_id =
        get<LinearSolver::Tags::IterationId>(box).step_number;

    if (next_orthogonalization_iteration_id <= iteration_id) {
      Parallel::contribute_to_reduction<
          StoreOrthogonalization<ParallelComponent>>(
          Parallel::ReductionData<
              Parallel::ReductionDatum<double, funcl::Plus<>>>{
              inner_product(gsl::at(get<basis_history_tag>(box),
                                    next_orthogonalization_iteration_id),
                            get<operand_tag>(box))},
          Parallel::get_parallel_component<ParallelComponent>(
              cache)[array_index],
          Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
              cache));
    } else {
      Parallel::contribute_to_reduction<
          StoreFinalOrthogonalization<ParallelComponent>>(
          Parallel::ReductionData<
              Parallel::ReductionDatum<double, funcl::Plus<>>>{
              inner_product(get<operand_tag>(box), get<operand_tag>(box))},
          Parallel::get_parallel_component<ParallelComponent>(
              cache)[array_index],
          Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
              cache));
    }
  }
};

struct NormalizeOperandAndUpdateField {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl2::flat_any_v<cpp17::is_same_v<
                db::add_tag_prefix<LinearSolver::Tags::Operand,
                                   typename Metavariables::system::fields_tag>,
                DbTags>...>> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double normalization,
                    const DenseVector<double>& minres,
                    const double residual_magnitude,
                    const bool has_converged) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using initial_fields_tag =
        db::add_tag_prefix<LinearSolver::Tags::Initial, fields_tag>;
    using operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
    using orthogonalization_iteration_id_tag =
        db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                           LinearSolver::Tags::IterationId>;
    using basis_history_tag =
        LinearSolver::Tags::KrylovSubspaceBasis<fields_tag>;
    using residual_magnitude_tag = db::add_tag_prefix<
        LinearSolver::Tags::Magnitude,
        db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;

    db::mutate<LinearSolver::Tags::IterationId,
               ::Tags::Next<LinearSolver::Tags::IterationId>,
               orthogonalization_iteration_id_tag, operand_tag,
               basis_history_tag, fields_tag, residual_magnitude_tag,
               LinearSolver::Tags::HasConverged>(
        make_not_null(&box),
        [
          normalization, minres, residual_magnitude, has_converged
        ](const gsl::not_null<IterationId*> iteration_id,
          const gsl::not_null<IterationId*> next_iteration_id,
          const gsl::not_null<IterationId*> orthogonalization_iteration_id,
          const gsl::not_null<db::item_type<operand_tag>*> operand,
          const gsl::not_null<db::item_type<basis_history_tag>*> basis_history,
          const gsl::not_null<db::item_type<fields_tag>*> field,
          const gsl::not_null<double*> local_residual_magnitude,
          const gsl::not_null<bool*> local_has_converged,
          const db::item_type<initial_fields_tag>& initial_field) noexcept {
          iteration_id->step_number++;
          next_iteration_id->step_number = iteration_id->step_number + 1;
          orthogonalization_iteration_id->step_number = 0;
          *operand /= normalization;
          basis_history->push_back(*operand);
          *field = initial_field;
          for (size_t i = 0; i < minres.size(); i++) {
            *field += minres[i] * gsl::at(*basis_history, i);
          }
          *local_residual_magnitude = residual_magnitude;
          *local_has_converged = has_converged;
        },
        get<initial_fields_tag>(box));

    // Proceed with algorithm
    // We use `ckLocal()` here since this is essentially retrieving "self",
    // which is guaranteed to be on the local processor. This ensures the calls
    // are evaluated in order.
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .ckLocal()
        ->set_terminate(false);
    Parallel::get_parallel_component<ParallelComponent>(cache)[array_index]
        .perform_algorithm();
  }
};

}  // namespace gmres_detail
}  // namespace LinearSolver
