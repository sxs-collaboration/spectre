// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Tags/InboxTags.hpp"
#include "ParallelAlgorithms/LinearSolver/Observe.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
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
      ::Tags::Initial<residual_magnitude_tag>;
  using previous_residual_magnitude_tag =
      ::Tags::Previous<residual_magnitude_tag>;
  using orthogonalization_history_tag =
      LinearSolver::Tags::OrthogonalizationHistory<fields_tag>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double residual_magnitude) {
    constexpr size_t iteration_id = 0;

    db::mutate<initial_residual_magnitude_tag, previous_residual_magnitude_tag>(
        [residual_magnitude](
            const gsl::not_null<double*> initial_residual_magnitude,
            const gsl::not_null<double*> previous_residual_magnitude) {
          *initial_residual_magnitude = residual_magnitude;
          *previous_residual_magnitude = residual_magnitude;
        },
        make_not_null(&box));

    LinearSolver::observe_detail::contribute_to_reduction_observer<
        OptionsGroup, ParallelComponent>(iteration_id, residual_magnitude,
                                         cache);

    // Determine whether the linear solver has already converged
    Convergence::HasConverged has_converged{
        get<Convergence::Tags::Criteria<OptionsGroup>>(box), iteration_id,
        residual_magnitude, residual_magnitude};

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
                 ::Verbosity::Quiet)) {
      Parallel::printf("%s initialized with residual: %e\n",
                       pretty_type::name<OptionsGroup>(), residual_magnitude);
    }
    if (UNLIKELY(has_converged and get<logging::Tags::Verbosity<OptionsGroup>>(
                                       cache) >= ::Verbosity::Quiet)) {
      Parallel::printf("%s has converged without any iterations: %s\n",
                       pretty_type::name<OptionsGroup>(), has_converged);
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
  using residual_magnitude_tag = LinearSolver::Tags::Magnitude<
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag =
      ::Tags::Initial<residual_magnitude_tag>;
  using previous_residual_magnitude_tag =
      ::Tags::Previous<residual_magnitude_tag>;
  using orthogonalization_history_tag =
      LinearSolver::Tags::OrthogonalizationHistory<fields_tag>;
  using ValueType =
      tt::get_complex_or_fundamental_type_t<typename fields_tag::type>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const size_t iteration_id,
                    const size_t orthogonalization_iteration_id,
                    const ValueType orthogonalization) {
    if (UNLIKELY(orthogonalization_iteration_id == 0)) {
      // Append a row and a column to the orthogonalization history. Zero the
      // entries that won't be set during the orthogonalization procedure below.
      db::mutate<orthogonalization_history_tag>(
          [iteration_id](const auto orthogonalization_history) {
            orthogonalization_history->resize(iteration_id + 1, iteration_id);
            for (size_t j = 0; j < orthogonalization_history->columns() - 1;
                 ++j) {
              (*orthogonalization_history)(
                  orthogonalization_history->rows() - 1, j) = 0.;
            }
          },
          make_not_null(&box));
    }

    // While the orthogonalization procedure is not complete, store the
    // orthogonalization, broadcast it back to all elements and return early
    if (orthogonalization_iteration_id < iteration_id) {
      db::mutate<orthogonalization_history_tag>(
          [orthogonalization, iteration_id, orthogonalization_iteration_id](
              const auto orthogonalization_history) {
            (*orthogonalization_history)(orthogonalization_iteration_id,
                                         iteration_id - 1) = orthogonalization;
          },
          make_not_null(&box));

      Parallel::receive_data<Tags::Orthogonalization<OptionsGroup, ValueType>>(
          Parallel::get_parallel_component<BroadcastTarget>(cache),
          iteration_id, orthogonalization);
      return;
    }

    // At this point, the orthogonalization procedure is complete.
    ASSERT(equal_within_roundoff(imag(orthogonalization), 0.0),
           "Normalization is not real: " << orthogonalization);
    const double normalization = sqrt(real(orthogonalization));
    db::mutate<orthogonalization_history_tag>(
        [normalization, iteration_id,
         orthogonalization_iteration_id](const auto orthogonalization_history) {
          (*orthogonalization_history)(orthogonalization_iteration_id,
                                       iteration_id - 1) = normalization;
        },
        make_not_null(&box));

    // Perform a QR decomposition of the Hessenberg matrix that was built during
    // the orthogonalization
    const auto& orthogonalization_history =
        get<orthogonalization_history_tag>(box);
    const auto num_rows = orthogonalization_iteration_id + 1;
    blaze::DynamicMatrix<ValueType> qr_Q;
    blaze::DynamicMatrix<ValueType> qr_R;
    blaze::qr(orthogonalization_history, qr_Q, qr_R);
    // Compute the residual vector from the QR decomposition
    blaze::DynamicVector<double> beta(num_rows, 0.);
    const double initial_residual_magnitude =
        get<initial_residual_magnitude_tag>(box);
    beta[0] = initial_residual_magnitude;
    blaze::DynamicVector<ValueType> minres =
        blaze::inv(qr_R) * blaze::ctrans(qr_Q) * beta;
    blaze::DynamicVector<ValueType> res =
        beta - orthogonalization_history * minres;
    const double residual_magnitude = sqrt(magnitude_square(res));

    // At this point, the iteration is complete. We proceed with observing,
    // logging and checking convergence before broadcasting back to the
    // elements.

    LinearSolver::observe_detail::contribute_to_reduction_observer<
        OptionsGroup, ParallelComponent>(iteration_id, residual_magnitude,
                                         cache);

    // Determine whether the linear solver has converged.
    // GMRES is guaranteed to decrease the residual monotonically, so an
    // increase in the residual is an error.
    const auto& convergence_criteria =
        get<Convergence::Tags::Criteria<OptionsGroup>>(box);
    const double previous_residual_magnitude =
        get<previous_residual_magnitude_tag>(box);
    auto has_converged =
        residual_magnitude < previous_residual_magnitude
            ? Convergence::HasConverged{convergence_criteria, iteration_id,
                                        residual_magnitude,
                                        initial_residual_magnitude}
            : Convergence::HasConverged{
                  Convergence::Reason::Error,
                  MakeString{} << std::scientific
                               << "Residual should decrease monotonically, but "
                                  "increased from "
                               << previous_residual_magnitude << " to "
                               << residual_magnitude << ".",
                  iteration_id};

    db::mutate<previous_residual_magnitude_tag>(
        [residual_magnitude](
            const gsl::not_null<double*> stored_previous_residual_magnitude) {
          *stored_previous_residual_magnitude = residual_magnitude;
        },
        make_not_null(&box));

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
                 ::Verbosity::Quiet)) {
      Parallel::printf("%s(%zu) iteration complete. Remaining residual: %e\n",
                       pretty_type::name<OptionsGroup>(), iteration_id,
                       residual_magnitude);
    }
    if (UNLIKELY(has_converged and get<logging::Tags::Verbosity<OptionsGroup>>(
                                       cache) >= ::Verbosity::Quiet)) {
      if (has_converged.reason() == Convergence::Reason::Error) {
        Parallel::printf("%s has encountered an error in iteration %zu: %s\n",
                         pretty_type::name<OptionsGroup>(), iteration_id,
                         has_converged.error_message());
      } else {
        Parallel::printf("%s has converged in %zu iterations: %s\n",
                         pretty_type::name<OptionsGroup>(), iteration_id,
                         has_converged);
      }
    }

    Parallel::receive_data<
        Tags::FinalOrthogonalization<OptionsGroup, ValueType>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), iteration_id,
        std::make_tuple(normalization, std::move(minres),
                        // NOLINTNEXTLINE(performance-move-const-arg)
                        std::move(has_converged)));
  }
};

}  // namespace LinearSolver::gmres::detail
