// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Informer/Verbosity.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Tags/InboxTags.hpp"
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
/// \endcond

namespace LinearSolver::gmres::detail {

template <typename FieldsTag, typename OptionsGroup, typename BroadcastTarget>
struct InitializeResidualMagnitude {
 private:
  using fields_tag = FieldsTag;
  using residual_magnitude_tag = LinearSolver::Tags::Magnitude<
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag =
      LinearSolver::Tags::Initial<residual_magnitude_tag>;
  using orthogonalization_iteration_id_tag =
      LinearSolver::Tags::Orthogonalization<
          LinearSolver::Tags::IterationId<OptionsGroup>>;
  using orthogonalization_history_tag =
      LinearSolver::Tags::OrthogonalizationHistory<fields_tag>;

 public:
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex, typename DataBox = db::DataBox<DbTagsList>,
      Requires<db::tag_is_retrievable_v<residual_magnitude_tag, DataBox>> =
          nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double residual_magnitude) noexcept {
    db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>,
               residual_magnitude_tag, initial_residual_magnitude_tag,
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
          *iteration_id = 0;
          *orthogonalization_iteration_id = 0;
          *orthogonalization_history = DenseMatrix<double>{2, 1, 0.};
        });

    LinearSolver::observe_detail::contribute_to_reduction_observer<
        FieldsTag, OptionsGroup>(box, cache);

    // Determine whether the linear solver has already converged. This invokes
    // the compute item.
    const auto& has_converged =
        db::get<LinearSolver::Tags::HasConverged<OptionsGroup>>(box);

    // Do some logging
    if (UNLIKELY(static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(cache)) >=
                 static_cast<int>(::Verbosity::Verbose))) {
      Parallel::printf("Linear solver '" +
                           Options::name<OptionsGroup>() +
                           "' initialized with residual: %e\n",
                       residual_magnitude);
    }
    if (UNLIKELY(has_converged and
                 static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(cache)) >=
                     static_cast<int>(::Verbosity::Quiet))) {
      Parallel::printf("The linear solver '" +
                           Options::name<OptionsGroup>() +
                           "' has converged without any iterations: %s\n",
                       has_converged);
    }

    Parallel::receive_data<Tags::InitialOrthogonalization<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
        std::make_tuple(residual_magnitude, has_converged));
  }
};

template <typename FieldsTag, typename OptionsGroup, typename BroadcastTarget>
struct StoreOrthogonalization {
 private:
  using fields_tag = FieldsTag;
  using residual_magnitude_tag = LinearSolver::Tags::Magnitude<
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag =
      LinearSolver::Tags::Initial<residual_magnitude_tag>;
  using orthogonalization_iteration_id_tag =
      LinearSolver::Tags::Orthogonalization<
          LinearSolver::Tags::IterationId<OptionsGroup>>;
  using orthogonalization_history_tag =
      LinearSolver::Tags::OrthogonalizationHistory<fields_tag>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<orthogonalization_history_tag,
                                              DataBox>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double orthogonalization) noexcept {
    // While the orthogonalization procedure is not complete, store the
    // orthogonalization, broadcast it back to all elements and return early
    const size_t iteration_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
    if (get<orthogonalization_iteration_id_tag>(box) <= iteration_id) {
      db::mutate<orthogonalization_history_tag,
                 orthogonalization_iteration_id_tag>(
          make_not_null(&box),
          [orthogonalization, iteration_id](
              const auto orthogonalization_history,
              const gsl::not_null<size_t*>
                  orthogonalization_iteration_id) noexcept {
            (*orthogonalization_history)(*orthogonalization_iteration_id,
                                         iteration_id) = orthogonalization;
            ++(*orthogonalization_iteration_id);
          });

      Parallel::receive_data<Tags::Orthogonalization<OptionsGroup>>(
          Parallel::get_parallel_component<BroadcastTarget>(cache),
          iteration_id, orthogonalization);
      return;
    }

    // At this point, the orthogonalization procedure is complete.
    db::mutate<orthogonalization_history_tag>(
        make_not_null(&box),
        [orthogonalization, iteration_id](
            const auto orthogonalization_history,
            const size_t& orthogonalization_iteration_id) noexcept {
          (*orthogonalization_history)(orthogonalization_iteration_id,
                                       iteration_id) = sqrt(orthogonalization);
        },
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
    DenseVector<double> minres = blaze::inv(qr_R) * blaze::trans(qr_Q) * beta;
    const double residual_magnitude =
        blaze::length(beta - orthogonalization_history * minres);

    // Store residual magnitude and prepare for the next iteration
    db::mutate<
        residual_magnitude_tag, LinearSolver::Tags::IterationId<OptionsGroup>,
        orthogonalization_iteration_id_tag, orthogonalization_history_tag>(
        make_not_null(&box),
        [residual_magnitude](
            const gsl::not_null<double*> local_residual_magnitude,
            const gsl::not_null<size_t*> local_iteration_id,
            const gsl::not_null<size_t*> orthogonalization_iteration_id,
            const gsl::not_null<typename orthogonalization_history_tag::type*>
                local_orthogonalization_history) noexcept {
          *local_residual_magnitude = residual_magnitude;
          // Prepare for the next iteration
          ++(*local_iteration_id);
          *orthogonalization_iteration_id = 0;
          local_orthogonalization_history->resize(*local_iteration_id + 2,
                                                  *local_iteration_id + 1);
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

    LinearSolver::observe_detail::contribute_to_reduction_observer<
        FieldsTag, OptionsGroup>(box, cache);

    // Determine whether the linear solver has converged. This invokes the
    // compute item.
    const auto& has_converged =
        db::get<LinearSolver::Tags::HasConverged<OptionsGroup>>(box);

    // Do some logging
    if (UNLIKELY(static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(cache)) >=
                 static_cast<int>(::Verbosity::Verbose))) {
      Parallel::printf("Linear solver '" +
                           Options::name<OptionsGroup>() +
                           "' iteration %zu done. Remaining residual: %e\n",
                       get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
                       residual_magnitude);
    }
    if (UNLIKELY(has_converged and
                 static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(cache)) >=
                     static_cast<int>(::Verbosity::Quiet))) {
      Parallel::printf("The linear solver '" +
                           Options::name<OptionsGroup>() +
                           "' has converged in %zu iterations: %s\n",
                       get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
                       has_converged);
    }

    Parallel::receive_data<Tags::FinalOrthogonalization<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), iteration_id,
        std::make_tuple(sqrt(orthogonalization), std::move(minres),
                        has_converged));
  }
};

}  // namespace LinearSolver::gmres::detail
