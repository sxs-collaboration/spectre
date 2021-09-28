// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/Tags/InboxTags.hpp"
#include "ParallelAlgorithms/LinearSolver/Observe.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace LinearSolver::cg::detail {

template <typename FieldsTag, typename OptionsGroup, typename BroadcastTarget>
struct InitializeResidual {
 private:
  using fields_tag = FieldsTag;
  using residual_square_tag = LinearSolver::Tags::MagnitudeSquare<
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag =
      ::Tags::Initial<LinearSolver::Tags::Magnitude<
          db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<residual_square_tag, DataBox>> =
                nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double residual_square) {
    constexpr size_t iteration_id = 0;
    const double residual_magnitude = sqrt(residual_square);

    db::mutate<residual_square_tag, initial_residual_magnitude_tag>(
        make_not_null(&box),
        [residual_square, residual_magnitude](
            const gsl::not_null<double*> local_residual_square,
            const gsl::not_null<double*> initial_residual_magnitude) {
          *local_residual_square = residual_square;
          *initial_residual_magnitude = residual_magnitude;
        });

    LinearSolver::observe_detail::contribute_to_reduction_observer<
        OptionsGroup, ParallelComponent>(iteration_id, residual_magnitude,
                                         cache);

    // Determine whether the linear solver has converged
    Convergence::HasConverged has_converged{
        get<Convergence::Tags::Criteria<OptionsGroup>>(box), iteration_id,
        residual_magnitude, residual_magnitude};

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
                 ::Verbosity::Quiet)) {
      Parallel::printf("%s initialized with residual: %e\n",
                       Options::name<OptionsGroup>(), residual_magnitude);
    }
    if (UNLIKELY(has_converged and get<logging::Tags::Verbosity<OptionsGroup>>(
                                       cache) >= ::Verbosity::Quiet)) {
      Parallel::printf(
          "%s has converged without any iterations: %s\n",
          Options::name<OptionsGroup>(), has_converged);
    }

    Parallel::receive_data<Tags::InitialHasConverged<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), iteration_id,
        // NOLINTNEXTLINE(performance-move-const-arg)
        std::move(has_converged));
  }
};

template <typename FieldsTag, typename OptionsGroup, typename BroadcastTarget>
struct ComputeAlpha {
 private:
  using fields_tag = FieldsTag;
  using residual_square_tag = LinearSolver::Tags::MagnitudeSquare<
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<residual_square_tag, DataBox>> =
                nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const size_t iteration_id,
                    const double conj_grad_inner_product) {
    Parallel::receive_data<Tags::Alpha<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), iteration_id,
        get<residual_square_tag>(box) / conj_grad_inner_product);
  }
};

template <typename FieldsTag, typename OptionsGroup, typename BroadcastTarget>
struct UpdateResidual {
 private:
  using fields_tag = FieldsTag;
  using residual_square_tag = LinearSolver::Tags::MagnitudeSquare<
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag =
      ::Tags::Initial<LinearSolver::Tags::Magnitude<
          db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<residual_square_tag, DataBox>> =
                nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const size_t iteration_id, const double residual_square) {
    // Compute the residual ratio before mutating the DataBox
    const double res_ratio = residual_square / get<residual_square_tag>(box);

    db::mutate<residual_square_tag>(
        make_not_null(&box),
        [residual_square](const gsl::not_null<double*> local_residual_square) {
          *local_residual_square = residual_square;
        });

    // At this point, the iteration is complete. We proceed with observing,
    // logging and checking convergence before broadcasting back to the
    // elements.

    const size_t completed_iterations = iteration_id + 1;
    const double residual_magnitude = sqrt(residual_square);
    LinearSolver::observe_detail::contribute_to_reduction_observer<
        OptionsGroup, ParallelComponent>(completed_iterations,
                                         residual_magnitude, cache);

    // Determine whether the linear solver has converged
    Convergence::HasConverged has_converged{
        get<Convergence::Tags::Criteria<OptionsGroup>>(box),
        completed_iterations, residual_magnitude,
        get<initial_residual_magnitude_tag>(box)};

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
                 ::Verbosity::Quiet)) {
      Parallel::printf(
          "%s(%zu) iteration complete. Remaining residual: %e\n",
          Options::name<OptionsGroup>(), completed_iterations,
          residual_magnitude);
    }
    if (UNLIKELY(has_converged and get<logging::Tags::Verbosity<OptionsGroup>>(
                                       cache) >= ::Verbosity::Quiet)) {
      Parallel::printf(
          "%s has converged in %zu iterations: %s\n",
          Options::name<OptionsGroup>(), completed_iterations, has_converged);
    }

    Parallel::receive_data<Tags::ResidualRatioAndHasConverged<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), iteration_id,
        // NOLINTNEXTLINE(performance-move-const-arg)
        std::make_tuple(res_ratio, std::move(has_converged)));
  }
};

}  // namespace LinearSolver::cg::detail
