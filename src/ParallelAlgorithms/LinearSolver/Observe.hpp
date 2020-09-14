// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/PrettyType.hpp"

namespace LinearSolver {
namespace observe_detail {

using reduction_data = Parallel::ReductionData<
    // Iteration
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Residual
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>>;

template <typename OptionsGroup>
struct Registration {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey{pretty_type::get_name<OptionsGroup>()}};
  }
};

/*!
 * \brief Contributes data from the residual monitor to the reduction observer
 *
 * With:
 * - `residual_magnitude_tag` = `
 * LinearSolver::Tags::Magnitude<db::add_tag_prefix<
 * LinearSolver::Tags::Residual, fields_tag>>`
 *
 * Uses:
 * - System:
 *   - `fields_tag`
 * - DataBox:
 *   - `LinearSolver::Tags::IterationId`
 *   - `residual_magnitude_tag`
 */
template <typename FieldsTag, typename OptionsGroup, typename DbTagsList,
          typename Metavariables>
void contribute_to_reduction_observer(
    db::DataBox<DbTagsList>& box,
    Parallel::GlobalCache<Metavariables>& cache) noexcept {
  using fields_tag = FieldsTag;
  using residual_magnitude_tag = LinearSolver::Tags::Magnitude<
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>;

  const auto observation_id = observers::ObservationId(
      get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
      pretty_type::get_name<OptionsGroup>());
  auto& reduction_writer = Parallel::get_parallel_component<
      observers::ObserverWriter<Metavariables>>(cache);
  Parallel::threaded_action<observers::ThreadedActions::WriteReductionData>(
      // Node 0 is always the writer, so directly call the component on that
      // node
      reduction_writer[0], observation_id,
      static_cast<size_t>(Parallel::my_node()),
      // When multiple linear solves are performed, e.g. for the nonlinear
      // solver, we'll need to write into separate subgroups, e.g.:
      // `/linear_residuals/<nonlinear_iteration_id>`
      std::string{"/" + Options::name<OptionsGroup>() + "Residuals"},
      std::vector<std::string>{"Iteration", "Residual"},
      reduction_data{get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
                     get<residual_magnitude_tag>(box)});
}

}  // namespace observe_detail
}  // namespace LinearSolver
