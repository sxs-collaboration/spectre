// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <variant>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/LineSearch.hpp"
#include "ParallelAlgorithms/NonlinearSolver/NewtonRaphson/Tags/InboxTags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Observe.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Functional.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace NonlinearSolver::newton_raphson::detail {

template <typename FieldsTag, typename OptionsGroup, typename BroadcastTarget>
struct CheckResidualMagnitude {
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>;
  using residual_magnitude_square_tag =
      LinearSolver::Tags::MagnitudeSquare<residual_tag>;
  using initial_residual_magnitude_tag =
      ::Tags::Initial<LinearSolver::Tags::Magnitude<residual_tag>>;
  using prev_residual_magnitude_square_tag =
      NonlinearSolver::Tags::Globalization<residual_magnitude_square_tag>;

  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex, typename... Args>
  static void apply(DataBox& box, Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, Args&&... args) {
    if constexpr (db::tag_is_retrievable_v<residual_magnitude_square_tag,
                                           DataBox>) {
      apply_impl<ParallelComponent>(box, cache, std::forward<Args>(args)...);
    } else {
      ERROR(
          "The residual monitor is not yet initialized. This is a bug, so "
          "please file an issue.");
    }
  }

 private:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables>
  static void apply_impl(db::DataBox<DbTagsList>& box,
                         Parallel::GlobalCache<Metavariables>& cache,
                         const size_t iteration_id,
                         const size_t globalization_iteration_id,
                         const double next_residual_magnitude_square,
                         const double step_length) {
    const double residual_magnitude = sqrt(next_residual_magnitude_square);

    NonlinearSolver::observe_detail::contribute_to_reduction_observer<
        OptionsGroup, ParallelComponent>(
        iteration_id, globalization_iteration_id, residual_magnitude,
        step_length, cache);

    if (UNLIKELY(iteration_id == 0)) {
      db::mutate<initial_residual_magnitude_tag>(
          make_not_null(&box),
          [residual_magnitude](
              const gsl::not_null<double*> initial_residual_magnitude) {
            *initial_residual_magnitude = residual_magnitude;
          });
    } else {
      // Make sure we are converging. Far away from the solution the correction
      // determined by the linear solve might be bad, so we employ a
      // globalization strategy to guide the solver towards the solution when
      // the residual doesn't decrease sufficiently. See the `NewtonRaphson`
      // class documentation for details.
      const double sufficient_decrease =
          get<NonlinearSolver::Tags::SufficientDecrease<OptionsGroup>>(box);
      const double residual_magnitude_square =
          get<residual_magnitude_square_tag>(box);
      const double initial_residual_magnitude =
          get<initial_residual_magnitude_tag>(box);
      const double abs_tolerance =
          get<Convergence::Tags::Criteria<OptionsGroup>>(box).absolute_residual;
      const double rel_tolerance =
          get<Convergence::Tags::Criteria<OptionsGroup>>(box).relative_residual;
      // This is the directional derivative of the residual magnitude square
      // f(x) = |r(x)|^2 in the descent direction
      const double residual_magnitude_square_slope =
          -2. * residual_magnitude_square;
      // Check the sufficient decrease condition. Also make sure the residual
      // didn't hit the tolerance.
      if (residual_magnitude > abs_tolerance and
          residual_magnitude / initial_residual_magnitude > rel_tolerance and
          next_residual_magnitude_square >
              residual_magnitude_square + sufficient_decrease * step_length *
                                              residual_magnitude_square_slope) {
        // The residual didn't sufficiently decrease. Perform a globalization
        // step.
        if (globalization_iteration_id <
            get<NonlinearSolver::Tags::MaxGlobalizationSteps<OptionsGroup>>(
                box)) {
          const double next_step_length = std::clamp(
              NonlinearSolver::newton_raphson::next_step_length(
                  globalization_iteration_id, step_length,
                  get<NonlinearSolver::Tags::StepLength<OptionsGroup>>(box),
                  residual_magnitude_square, residual_magnitude_square_slope,
                  next_residual_magnitude_square,
                  get<prev_residual_magnitude_square_tag>(box)),
              0.1 * step_length, 0.5 * step_length);
          db::mutate<NonlinearSolver::Tags::StepLength<OptionsGroup>,
                     prev_residual_magnitude_square_tag>(
              make_not_null(&box),
              [step_length, next_residual_magnitude_square](
                  const gsl::not_null<double*> prev_step_length,
                  const gsl::not_null<double*> prev_residual_magnitude_square) {
                *prev_step_length = step_length;
                *prev_residual_magnitude_square =
                    next_residual_magnitude_square;
              });
          // Do some logging
          if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                       ::Verbosity::Verbose)) {
            Parallel::printf(
                "%s(%zu): Step with length %g didn't sufficiently decrease the "
                "residual (possible overshoot). Residual: %e. Next step "
                "length: %g.\n",
                Options::name<OptionsGroup>(), iteration_id, step_length,
                residual_magnitude, next_step_length);
            if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                         ::Verbosity::Debug)) {
              Parallel::printf("Residual magnitude slope: %e\n",
                               residual_magnitude_square_slope);
            }
          }
          // Broadcast back to the elements signaling that they should perform a
          // globalization step, then return early.
          Parallel::receive_data<Tags::GlobalizationResult<OptionsGroup>>(
              Parallel::get_parallel_component<BroadcastTarget>(cache),
              iteration_id,
              std::variant<double, Convergence::HasConverged>{
                  next_step_length});
          return;
        } else if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                            ::Verbosity::Quiet)) {
          Parallel::printf(
              "%s(%zu): WARNING: Failed to sufficiently decrease the residual "
              "in %zu globalization steps. This is usually indicative of an "
              "ill-posed problem, for example when the linearization of the "
              "nonlinear operator is not computed correctly.",
              Options::name<OptionsGroup>(), iteration_id,
              globalization_iteration_id);
        }  // min_step_length
      }    // sufficient decrease condition
    }      // initial iteration

    db::mutate<residual_magnitude_square_tag>(
        make_not_null(&box),
        [next_residual_magnitude_square](
            const gsl::not_null<double*> local_residual_magnitude_square) {
          *local_residual_magnitude_square = next_residual_magnitude_square;
        });

    // At this point, the iteration is complete. We proceed with logging and
    // checking convergence before broadcasting back to the elements.

    // Determine whether the nonlinear solver has converged
    Convergence::HasConverged has_converged{
        get<Convergence::Tags::Criteria<OptionsGroup>>(box), iteration_id,
        residual_magnitude, get<initial_residual_magnitude_tag>(box)};

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Quiet)) {
      if (UNLIKELY(iteration_id == 0)) {
        Parallel::printf("%s initialized with residual: %e\n",
                         Options::name<OptionsGroup>(), residual_magnitude);
      } else {
        Parallel::printf(
            "%s(%zu) iteration complete (%zu globalization steps, step length "
            "%g). Remaining residual: %e\n",
            Options::name<OptionsGroup>(), iteration_id,
            globalization_iteration_id, step_length, residual_magnitude);
      }
    }
    if (UNLIKELY(has_converged and get<logging::Tags::Verbosity<OptionsGroup>>(
                                       box) >= ::Verbosity::Quiet)) {
      if (UNLIKELY(iteration_id == 0)) {
        Parallel::printf("%s has converged without any iterations: %s\n",
                         Options::name<OptionsGroup>(), has_converged);
      } else {
        Parallel::printf("%s has converged in %zu iterations: %s\n",
                         Options::name<OptionsGroup>(), iteration_id,
                         has_converged);
      }
    }

    Parallel::receive_data<Tags::GlobalizationResult<OptionsGroup>>(
        Parallel::get_parallel_component<BroadcastTarget>(cache), iteration_id,
        std::variant<double, Convergence::HasConverged>(
            // NOLINTNEXTLINE(performance-move-const-arg)
            std::move(has_converged)));
  }
};

}  // namespace NonlinearSolver::newton_raphson::detail
