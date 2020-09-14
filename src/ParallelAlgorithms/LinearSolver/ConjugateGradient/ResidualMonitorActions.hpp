// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
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
  using residual_magnitude_tag = LinearSolver::Tags::Magnitude<
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
  using initial_residual_magnitude_tag =
      LinearSolver::Tags::Initial<residual_magnitude_tag>;

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<residual_square_tag, DataBox>> =
                nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double residual_square) noexcept {
    db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>,
               residual_square_tag>(
        make_not_null(&box),
        [residual_square](
            const gsl::not_null<size_t*> iteration_id,
            const gsl::not_null<double*> local_residual_square) noexcept {
          *local_residual_square = residual_square;
          *iteration_id = 0;
        });
    // Perform a separate `db::mutate` so that we can retrieve the
    // `residual_magnitude_tag` from the compute item
    db::mutate<initial_residual_magnitude_tag>(
        make_not_null(&box),
        [](const gsl::not_null<double*> local_initial_residual_magnitude,
           const double initial_residual_magnitude) noexcept {
          *local_initial_residual_magnitude = initial_residual_magnitude;
        },
        get<residual_magnitude_tag>(box));

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
                           "' initialized with residual: %e\n",
                       get<residual_magnitude_tag>(box));
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

    Parallel::receive_data<Tags::InitialHasConverged<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box), has_converged);
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
                    const double conj_grad_inner_product) noexcept {
    Parallel::receive_data<Tags::Alpha<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache),
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
        get<residual_square_tag>(box) / conj_grad_inner_product);
  }
};

template <typename FieldsTag, typename OptionsGroup, typename BroadcastTarget>
struct UpdateResidual {
 private:
  using fields_tag = FieldsTag;
  using residual_square_tag = LinearSolver::Tags::MagnitudeSquare<
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;
  using residual_magnitude_tag = LinearSolver::Tags::Magnitude<
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
                    const double residual_square) noexcept {
    // Compute the residual ratio before mutating the DataBox
    const double res_ratio = residual_square / get<residual_square_tag>(box);
    const size_t iteration_id =
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);

    db::mutate<residual_square_tag,
               LinearSolver::Tags::IterationId<OptionsGroup>>(
        make_not_null(&box),
        [residual_square](
            const gsl::not_null<double*> local_residual_square,
            const gsl::not_null<size_t*> local_iteration_id) noexcept {
          *local_residual_square = residual_square;
          // Prepare for the next iteration
          ++(*local_iteration_id);
        });

    // At this point, the iteration is complete. We proceed with observing,
    // logging and checking convergence before broadcasting back to the
    // elements.

    LinearSolver::observe_detail::contribute_to_reduction_observer<
        FieldsTag, OptionsGroup>(box, cache);

    // Determine whether the linear solver has converged. This invokes the
    // compute item.
    const auto& has_converged =
        get<LinearSolver::Tags::HasConverged<OptionsGroup>>(box);

    // Do some logging
    if (UNLIKELY(static_cast<int>(
                     get<LinearSolver::Tags::Verbosity<OptionsGroup>>(cache)) >=
                 static_cast<int>(::Verbosity::Verbose))) {
      Parallel::printf("Linear solver '" +
                           Options::name<OptionsGroup>() +
                           "' iteration %zu done. Remaining residual: %e\n",
                       get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
                       get<residual_magnitude_tag>(box));
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

    Parallel::receive_data<Tags::ResidualRatioAndHasConverged<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), iteration_id,
        std::make_tuple(res_ratio, has_converged));
  }
};

}  // namespace LinearSolver::cg::detail
