// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

/// Functionality shared between parallel linear solvers that have no global
/// synchronization points
namespace LinearSolver::async_solvers {

using reduction_data = Parallel::ReductionData<
    // Iteration
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Residual
    Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>;

template <typename OptionsGroup>
struct ElementObservationType {};

template <typename FieldsTag, typename OptionsGroup, typename DbTagsList,
          typename Metavariables, typename ArrayIndex>
void contribute_to_residual_observation(
    const db::DataBox<DbTagsList>& box,
    Parallel::GlobalCache<Metavariables>& cache,
    const ArrayIndex& array_index) noexcept {
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>;
  using residual_magnitude_square_tag =
      LinearSolver::Tags::MagnitudeSquare<residual_tag>;

  const size_t iteration_id =
      get<LinearSolver::Tags::IterationId<OptionsGroup>>(box);
  const double residual_magnitude_square =
      get<residual_magnitude_square_tag>(box);
  auto& local_observer =
      *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
           cache)
           .ckLocalBranch();
  Parallel::simple_action<observers::Actions::ContributeReductionData>(
      local_observer,
      observers::ObservationId(iteration_id,
                               ElementObservationType<OptionsGroup>{}),
      std::string{"/" + option_name<OptionsGroup>() + "Residuals"},
      std::vector<std::string>{"Iteration", "Residual"},
      reduction_data{iteration_id, residual_magnitude_square});
  if (UNLIKELY(static_cast<int>(
                   get<LinearSolver::Tags::Verbosity<OptionsGroup>>(cache)) >=
               static_cast<int>(::Verbosity::Verbose))) {
    if (iteration_id == 0) {
      Parallel::printf(
          "Linear solver '" + option_name<OptionsGroup>() +
              "' initialized on element %s. Remaining local residual: %e\n",
          get_output(array_index), sqrt(residual_magnitude_square));
    } else {
      Parallel::printf("Linear solver '" + option_name<OptionsGroup>() +
                           "' iteration %zu done on element %s. Remaining "
                           "local residual: %e\n",
                       iteration_id, get_output(array_index),
                       sqrt(residual_magnitude_square));
    }
  }
}

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using source_tag = SourceTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  using residual_magnitude_square_tag =
      LinearSolver::Tags::MagnitudeSquare<residual_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Tags::Iterations<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<LinearSolver::Tags::IterationId<OptionsGroup>,
                              residual_magnitude_square_tag,
                              operator_applied_to_fields_tag>,
            db::AddComputeTags<
                LinearSolver::Tags::ResidualCompute<fields_tag, source_tag>,
                LinearSolver::Tags::HasConvergedByIterationsCompute<
                    OptionsGroup>>>(
            std::move(box),
            // The `PrepareSolve` action populates these tags with initial
            // values, except for `operator_applied_to_fields_tag` which is
            // expected to be updated in every iteration of the algorithm
            std::numeric_limits<size_t>::max(),
            std::numeric_limits<double>::signaling_NaN(),
            typename operator_applied_to_fields_tag::type{}));
  }
};

template <typename OptionsGroup>
struct RegisterObservers {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationId>
  register_info(const db::DataBox<DbTagsList>& box,
                const ArrayIndex& /*array_index*/) noexcept {
    return {
        observers::TypeOfObservation::Reduction,
        observers::ObservationId{
            static_cast<double>(
                db::get<LinearSolver::Tags::IterationId<OptionsGroup>>(box)),
            ElementObservationType<OptionsGroup>{}}};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
using RegisterElement =
    observers::Actions::RegisterWithObservers<RegisterObservers<OptionsGroup>>;

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>;
  using residual_magnitude_square_tag =
      LinearSolver::Tags::MagnitudeSquare<residual_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>,
               residual_magnitude_square_tag>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id,
           const gsl::not_null<double*> residual_magnitude_square,
           const auto& residual) noexcept {
          *iteration_id = 0;
          *residual_magnitude_square = inner_product(residual, residual);
        },
        get<residual_tag>(box));
    // Observe the initial residual even if no steps are going to be performed
    contribute_to_residual_observation<FieldsTag, OptionsGroup>(box, cache,
                                                                array_index);
    return {std::move(box)};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct CompleteStep {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  using residual_magnitude_square_tag =
      LinearSolver::Tags::MagnitudeSquare<residual_tag>;

 public:
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Tags::Verbosity<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Update and observe element-local residual magnitude
    db::mutate<residual_magnitude_square_tag,
               LinearSolver::Tags::IterationId<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<double*> residual_magnitude_square,
           const gsl::not_null<size_t*> iteration_id,
           const auto& residual) noexcept {
          *residual_magnitude_square = inner_product(residual, residual);
          ++(*iteration_id);
        },
        get<residual_tag>(box));
    contribute_to_residual_observation<FieldsTag, OptionsGroup>(box, cache,
                                                                array_index);
    return {std::move(box)};
  }
};

}  // namespace LinearSolver::async_solvers
