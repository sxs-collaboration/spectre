// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/LinearSolver/Observe.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
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
template <typename FieldsTag>
struct NormalizeInitialOperand;
template <typename FieldsTag>
struct OrthogonalizeOperand;
template <typename FieldsTag>
struct NormalizeOperandAndUpdateField;
}  // namespace gmres_detail
}  // namespace LinearSolver
/// \endcond

namespace LinearSolver {
namespace gmres_detail {

template <typename FieldsTag, typename BroadcastTarget>
struct InitializeResidualMagnitude {
 private:
  using fields_tag = FieldsTag;
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

 public:
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex, typename DataBox = db::DataBox<DbTagsList>,
      Requires<db::tag_is_retrievable_v<residual_magnitude_tag, DataBox>> =
          nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double residual_magnitude) noexcept {
    db::mutate<LinearSolver::Tags::IterationId, residual_magnitude_tag,
               initial_residual_magnitude_tag,
               orthogonalization_iteration_id_tag,
               orthogonalization_history_tag>(
        make_not_null(&box),
        [residual_magnitude](
            const gsl::not_null<size_t*> iteration_id,
            const gsl::not_null<double*> local_residual_magnitude,
            const gsl::not_null<double*> initial_residual_magnitude,
            const gsl::not_null<size_t*> orthogonalization_iteration_id,
            const gsl::not_null<DenseMatrix<double>*>
                orthogonalization_history) noexcept {
          *local_residual_magnitude = *initial_residual_magnitude =
              residual_magnitude;
          // Also setting the following tags so re-initialization works:
          // - LinearSolver::Tags::IterationId
          // - orthogonalization_iteration_id_tag
          // - orthogonalization_history_tag
          *iteration_id = 0;
          *orthogonalization_iteration_id = 0;
          *orthogonalization_history = DenseMatrix<double>{2, 1, 0.};
        });

    LinearSolver::observe_detail::contribute_to_reduction_observer<FieldsTag>(
        box, cache);

    // Determine whether the linear solver has already converged. This invokes
    // the compute item.
    const auto& has_converged = db::get<LinearSolver::Tags::HasConverged>(box);

    // Do some logging
    if (UNLIKELY(has_converged and
                 static_cast<int>(get<LinearSolver::Tags::Verbosity>(cache)) >=
                     static_cast<int>(::Verbosity::Quiet))) {
      Parallel::printf(
          "The linear solver has converged without any iterations: %s\n",
          has_converged);
    }

    Parallel::simple_action<NormalizeInitialOperand<FieldsTag>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        residual_magnitude, has_converged);
  }
};

template <typename FieldsTag, typename BroadcastTarget>
struct StoreOrthogonalization {
 private:
  using fields_tag = FieldsTag;
  using orthogonalization_iteration_id_tag =
      db::add_tag_prefix<LinearSolver::Tags::Orthogonalization,
                         LinearSolver::Tags::IterationId>;
  using orthogonalization_history_tag =
      db::add_tag_prefix<LinearSolver::Tags::OrthogonalizationHistory,
                         fields_tag>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<orthogonalization_history_tag,
                                              DataBox>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double orthogonalization) noexcept {
    db::mutate<orthogonalization_history_tag,
               orthogonalization_iteration_id_tag>(
        make_not_null(&box),
        [orthogonalization](
            const gsl::not_null<db::item_type<orthogonalization_history_tag>*>
                orthogonalization_history,
            const gsl::not_null<
                db::item_type<orthogonalization_iteration_id_tag>*>
                orthogonalization_iteration_id,
            const db::const_item_type<LinearSolver::Tags::IterationId>&
                iteration_id) noexcept {
          (*orthogonalization_history)(*orthogonalization_iteration_id,
                                       iteration_id) = orthogonalization;
          (*orthogonalization_iteration_id)++;
        },
        get<LinearSolver::Tags::IterationId>(box));

    Parallel::simple_action<OrthogonalizeOperand<FieldsTag>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        orthogonalization);
  }
};

template <typename FieldsTag, typename BroadcastTarget>
struct StoreFinalOrthogonalization {
 private:
  using fields_tag = FieldsTag;
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

 public:
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex, typename DataBox = db::DataBox<DbTagsList>,
      Requires<db::tag_is_retrievable_v<residual_magnitude_tag, DataBox>> =
          nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double orthogonalization) noexcept {
    db::mutate<orthogonalization_history_tag>(
        make_not_null(&box),
        [orthogonalization](
            const gsl::not_null<db::item_type<orthogonalization_history_tag>*>
                orthogonalization_history,
            const db::const_item_type<LinearSolver::Tags::IterationId>&
                iteration_id,
            const db::const_item_type<orthogonalization_iteration_id_tag>&
                orthogonalization_iteration_id) noexcept {
          (*orthogonalization_history)(orthogonalization_iteration_id,
                                       iteration_id) = sqrt(orthogonalization);
        },
        get<LinearSolver::Tags::IterationId>(box),
        get<orthogonalization_iteration_id_tag>(box));

    // Perform a QR decomposition of the Hessenberg matrix that was built during
    // the orthogonalization
    const auto& orthogonalization_history =
        get<orthogonalization_history_tag>(box);
    const auto num_rows = get<orthogonalization_iteration_id_tag>(box) + 1;
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
            const gsl::not_null<db::item_type<LinearSolver::Tags::IterationId>*>
                iteration_id,
            const gsl::not_null<
                db::item_type<orthogonalization_iteration_id_tag>*>
                orthogonalization_iteration_id,
            const gsl::not_null<db::item_type<orthogonalization_history_tag>*>
                local_orthogonalization_history) noexcept {
          *local_residual_magnitude = residual_magnitude;
          // Prepare for the next iteration
          (*iteration_id)++;
          *orthogonalization_iteration_id = 0;
          local_orthogonalization_history->resize(*iteration_id + 2,
                                                  *iteration_id + 1);
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

    LinearSolver::observe_detail::contribute_to_reduction_observer<FieldsTag>(
        box, cache);

    // Determine whether the linear solver has converged. This invokes the
    // compute item.
    const auto& has_converged = db::get<LinearSolver::Tags::HasConverged>(box);

    // Do some logging
    if (UNLIKELY(static_cast<int>(get<LinearSolver::Tags::Verbosity>(cache)) >=
                 static_cast<int>(::Verbosity::Verbose))) {
      Parallel::printf(
          "Linear solver iteration %zu done. Remaining residual: %e\n",
          get<LinearSolver::Tags::IterationId>(box), residual_magnitude);
    }
    if (UNLIKELY(has_converged and
                 static_cast<int>(get<LinearSolver::Tags::Verbosity>(cache)) >=
                     static_cast<int>(::Verbosity::Quiet))) {
      Parallel::printf(
          "The linear solver has converged in %zu iterations: %s\n",
          get<LinearSolver::Tags::IterationId>(box), has_converged);
    }

    Parallel::simple_action<NormalizeOperandAndUpdateField<FieldsTag>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        sqrt(orthogonalization), minres, has_converged);
  }
};

}  // namespace gmres_detail
}  // namespace LinearSolver
