// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace gmres_detail {
struct NormalizeInitialOperand;
struct OrthogonalizeOperand;
struct NormalizeOperandAndUpdateField;
}  // namespace gmres_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace gmres_detail {

template <typename BroadcastTarget>
struct InitializeResidualMagnitude {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double residual_magnitude) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using residual_magnitude_tag = db::add_tag_prefix<
        LinearSolver::Tags::Magnitude,
        db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
    using initial_residual_magnitude_tag =
        db::add_tag_prefix<LinearSolver::Tags::Initial, residual_magnitude_tag>;

    db::mutate<residual_magnitude_tag, initial_residual_magnitude_tag>(
        make_not_null(&box), [residual_magnitude](
                                 const gsl::not_null<double*>
                                     local_residual_magnitude,
                                 const gsl::not_null<double*>
                                     initial_residual_magnitude) noexcept {
          *local_residual_magnitude = *initial_residual_magnitude =
              residual_magnitude;
        });

    // Determine whether the linear solver has already converged. This invokes
    // the compute item.
    const auto& has_converged = db::get<LinearSolver::Tags::HasConverged>(box);

    if (UNLIKELY(has_converged and
                 static_cast<int>(get<::Tags::Verbosity>(box)) >=
                     static_cast<int>(::Verbosity::Quiet))) {
      Parallel::printf("%s", has_converged);
    }

    Parallel::simple_action<NormalizeInitialOperand>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        residual_magnitude, has_converged);
  }
};

template <typename BroadcastTarget>
struct StoreOrthogonalization {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double orthogonalization) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using orthogonalization_iteration_id_tag =
        db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                           LinearSolver::Tags::IterationId>;
    using orthogonalization_history_tag =
        db::add_tag_prefix<LinearSolver::Tags::OrthogonalizationHistory,
                           fields_tag>;

    db::mutate<orthogonalization_history_tag,
               orthogonalization_iteration_id_tag>(
        make_not_null(&box),
        [orthogonalization](
            const gsl::not_null<db::item_type<orthogonalization_history_tag>*>
                orthogonalization_history,
            const gsl::not_null<IterationId*> orthogonalization_iteration_id,
            const IterationId& iteration_id) noexcept {
          (*orthogonalization_history)(
              orthogonalization_iteration_id->step_number,
              iteration_id.step_number) = orthogonalization;
          orthogonalization_iteration_id->step_number++;
        },
        get<LinearSolver::Tags::IterationId>(box));

    Parallel::simple_action<OrthogonalizeOperand>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        orthogonalization);
  }
};

using observed_reduction_data = Parallel::ReductionData<
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>>;

template <typename BroadcastTarget>
struct StoreFinalOrthogonalization {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const double orthogonalization) noexcept {
    using fields_tag = typename Metavariables::system::fields_tag;
    using residual_magnitude_tag = db::add_tag_prefix<
        LinearSolver::Tags::Magnitude,
        db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
    using initial_residual_magnitude_tag =
        db::add_tag_prefix<LinearSolver::Tags::Initial, residual_magnitude_tag>;
    using orthogonalization_iteration_id_tag =
        db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                           LinearSolver::Tags::IterationId>;
    using orthogonalization_history_tag =
        db::add_tag_prefix<LinearSolver::Tags::OrthogonalizationHistory,
                           fields_tag>;

    db::mutate<orthogonalization_history_tag>(
        make_not_null(&box),
        [orthogonalization](
            const gsl::not_null<db::item_type<orthogonalization_history_tag>*>
                orthogonalization_history,
            const IterationId& iteration_id,
            const IterationId& orthogonalization_iteration_id) noexcept {
          (*orthogonalization_history)(
              orthogonalization_iteration_id.step_number,
              iteration_id.step_number) = sqrt(orthogonalization);
        },
        get<LinearSolver::Tags::IterationId>(box),
        get<orthogonalization_iteration_id_tag>(box));

    // Perform a QR decomposition of the Hessenberg matrix that was built during
    // the orthogonalization
    const auto& orthogonalization_history =
        get<orthogonalization_history_tag>(box);
    const auto num_rows =
        get<orthogonalization_iteration_id_tag>(box).step_number + 1;
    DenseMatrix<double> qr_Q;
    DenseMatrix<double> qr_R;
    blaze::qr(orthogonalization_history, qr_Q, qr_R);
    // Compute the residual vector from the QR decomposition
    DenseVector<double> beta(num_rows, 0.);
    beta[0] = get<initial_residual_magnitude_tag>(box);
    const DenseVector<double> minres =
        blaze::inv(qr_R) * blaze::trans(qr_Q) * beta;
    const double residual_magnitude =
        blaze::length(beta - orthogonalization_history * minres);

    // Store residual magnitude and prepare for the next iteration
    db::mutate<residual_magnitude_tag, LinearSolver::Tags::IterationId,
               orthogonalization_iteration_id_tag,
               orthogonalization_history_tag>(
        make_not_null(&box),
        [residual_magnitude](
            const gsl::not_null<double*> local_residual_magnitude,
            const gsl::not_null<IterationId*> iteration_id,
            const gsl::not_null<IterationId*> orthogonalization_iteration_id,
            const gsl::not_null<db::item_type<orthogonalization_history_tag>*>
                local_orthogonalization_history) noexcept {
          *local_residual_magnitude = residual_magnitude;
          // Prepare for the next iteration
          iteration_id->step_number++;
          orthogonalization_iteration_id->step_number = 0;
          local_orthogonalization_history->resize(
              iteration_id->step_number + 2, iteration_id->step_number + 1);
          // Make sure the new entries are zero
          for (size_t i = 0; i < local_orthogonalization_history->rows(); i++) {
            (*local_orthogonalization_history)(
                i, local_orthogonalization_history->columns() - 1) = 0.;
          }
          for (size_t j = 0; j < local_orthogonalization_history->columns();
               j++) {
            (*local_orthogonalization_history)(
                local_orthogonalization_history->rows() - 1, j) = 0.;
          }
        });

    // At this point, the iteration is complete. We proceed with observing,
    // logging and checking convergence before broadcasting back to the
    // elements.

    // Contribute data to the observer
    const auto observation_id =
        observers::ObservationId(get<LinearSolver::Tags::IterationId>(box));
    auto& reduction_writer = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);
    Parallel::threaded_action<observers::ThreadedActions::WriteReductionData>(
        // Node 0 is always the writer, so directly call the component on that
        // node
        reduction_writer[0], observation_id,
        // When multiple linear solves are performed, e.g. for the nonlinear
        // solver, we'll need to write into separate subgroups, e.g.:
        // `/linear_residuals/<nonlinear_iteration_id>`
        std::string{"/linear_residuals"},
        std::vector<std::string>{"Iteration", "Residual"},
        observed_reduction_data{
            get<LinearSolver::Tags::IterationId>(box).step_number,
            residual_magnitude});

    // Determine whether the linear solver has converged. This invokes the
    // compute item.
    const auto& has_converged = db::get<LinearSolver::Tags::HasConverged>(box);

    // Do some logging
    if (UNLIKELY(static_cast<int>(get<::Tags::Verbosity>(box)) >=
                 static_cast<int>(::Verbosity::Verbose))) {
      Parallel::printf(
          "Linear solver iteration %d done. Remaining residual: %e\n",
          get<LinearSolver::Tags::IterationId>(box).step_number,
          residual_magnitude);
    }
    if (UNLIKELY(has_converged and
                 static_cast<int>(get<::Tags::Verbosity>(box)) >=
                     static_cast<int>(::Verbosity::Quiet))) {
      Parallel::printf("%s", has_converged);
    }

    Parallel::simple_action<NormalizeOperandAndUpdateField>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        sqrt(orthogonalization), minres, has_converged);
  }
};

}  // namespace gmres_detail
}  // namespace LinearSolver
