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
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
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
  using initial_residual_magnitude_tag =
      ::Tags::Initial<LinearSolver::Tags::Magnitude<
          db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>>;
  using orthogonalization_history_tag =
      LinearSolver::Tags::OrthogonalizationHistory<fields_tag>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<initial_residual_magnitude_tag,
                                              DataBox>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double residual_magnitude) noexcept {
    constexpr size_t iteration_id = 0;

    db::mutate<initial_residual_magnitude_tag>(
        make_not_null(&box),
        [residual_magnitude](
            const gsl::not_null<double*> initial_residual_magnitude) noexcept {
          *initial_residual_magnitude = residual_magnitude;
        });

    LinearSolver::observe_detail::contribute_to_reduction_observer<
        OptionsGroup>(iteration_id, residual_magnitude, cache);

    // Determine whether the linear solver has already converged
    Convergence::HasConverged has_converged{
        get<Convergence::Tags::Criteria<OptionsGroup>>(box),
        iteration_id, residual_magnitude, residual_magnitude};

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
                 ::Verbosity::Verbose)) {
      Parallel::printf("Linear solver '" +
                           Options::name<OptionsGroup>() +
                           "' initialized with residual: %e\n",
                       residual_magnitude);
    }
    if (UNLIKELY(has_converged and
                 get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
                     ::Verbosity::Quiet)) {
      Parallel::printf("The linear solver '" +
                           Options::name<OptionsGroup>() +
                           "' has converged without any iterations: %s\n",
                       has_converged);
    }

    Parallel::receive_data<Tags::InitialOrthogonalization<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), iteration_id,
        // NOLINTNEXTLINE(performance-move-const-arg)
        std::make_tuple(residual_magnitude, std::move(has_converged)));
  }
};

template <typename FieldsTag, typename OptionsGroup, typename BroadcastTarget>
struct StoreOrthogonalization {
 private:
  using fields_tag = FieldsTag;
  using initial_residual_magnitude_tag =
      ::Tags::Initial<LinearSolver::Tags::Magnitude<
          db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>>;
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
                    const size_t iteration_id,
                    const size_t orthogonalization_iteration_id,
                    const double orthogonalization) noexcept {
    if (UNLIKELY(orthogonalization_iteration_id == 0)) {
      // Append a row and a column to the orthogonalization history. Zero the
      // entries that won't be set during the orthogonalization procedure below.
      db::mutate<orthogonalization_history_tag>(
          make_not_null(&box),
          [iteration_id](const auto orthogonalization_history) noexcept {
            orthogonalization_history->resize(iteration_id + 2,
                                              iteration_id + 1);
            for (size_t j = 0; j < orthogonalization_history->columns() - 1;
                 ++j) {
              (*orthogonalization_history)(
                  orthogonalization_history->rows() - 1, j) = 0.;
            }
          });
    }

    // While the orthogonalization procedure is not complete, store the
    // orthogonalization, broadcast it back to all elements and return early
    if (orthogonalization_iteration_id <= iteration_id) {
      db::mutate<orthogonalization_history_tag>(
          make_not_null(&box),
          [orthogonalization, iteration_id, orthogonalization_iteration_id](
              const auto orthogonalization_history) noexcept {
            (*orthogonalization_history)(orthogonalization_iteration_id,
                                         iteration_id) = orthogonalization;
          });

      Parallel::receive_data<Tags::Orthogonalization<OptionsGroup>>(
          Parallel::get_parallel_component<BroadcastTarget>(cache),
          iteration_id, orthogonalization);
      return;
    }

    // At this point, the orthogonalization procedure is complete.
    db::mutate<orthogonalization_history_tag>(
        make_not_null(&box),
        [orthogonalization, iteration_id, orthogonalization_iteration_id](
            const auto orthogonalization_history) noexcept {
          (*orthogonalization_history)(orthogonalization_iteration_id,
                                       iteration_id) = sqrt(orthogonalization);
        });

    // Perform a QR decomposition of the Hessenberg matrix that was built during
    // the orthogonalization
    const auto& orthogonalization_history =
        get<orthogonalization_history_tag>(box);
    const auto num_rows = orthogonalization_iteration_id + 1;
    DenseMatrix<double> qr_Q;
    DenseMatrix<double> qr_R;
    blaze::qr(orthogonalization_history, qr_Q, qr_R);
    // Compute the residual vector from the QR decomposition
    DenseVector<double> beta(num_rows, 0.);
    beta[0] = get<initial_residual_magnitude_tag>(box);
    DenseVector<double> minres = blaze::inv(qr_R) * blaze::trans(qr_Q) * beta;
    const double residual_magnitude =
        blaze::length(beta - orthogonalization_history * minres);

    // At this point, the iteration is complete. We proceed with observing,
    // logging and checking convergence before broadcasting back to the
    // elements.

    const size_t completed_iterations = iteration_id + 1;
    LinearSolver::observe_detail::contribute_to_reduction_observer<
        OptionsGroup>(completed_iterations, residual_magnitude, cache);

    // Determine whether the linear solver has converged
    Convergence::HasConverged has_converged{
        get<Convergence::Tags::Criteria<OptionsGroup>>(box),
        completed_iterations, residual_magnitude,
        get<initial_residual_magnitude_tag>(box)};

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
                 ::Verbosity::Verbose)) {
      Parallel::printf("Linear solver '" +
                           Options::name<OptionsGroup>() +
                           "' iteration %zu done. Remaining residual: %e\n",
                       completed_iterations, residual_magnitude);
    }
    if (UNLIKELY(has_converged and
                     get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
                     ::Verbosity::Quiet)) {
      Parallel::printf("The linear solver '" +
                           Options::name<OptionsGroup>() +
                           "' has converged in %zu iterations: %s\n",
                       completed_iterations, has_converged);
    }

    Parallel::receive_data<Tags::FinalOrthogonalization<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), iteration_id,
        std::make_tuple(sqrt(orthogonalization), std::move(minres),
                        // NOLINTNEXTLINE(performance-move-const-arg)
                        std::move(has_converged)));
  }
};

}  // namespace LinearSolver::gmres::detail
