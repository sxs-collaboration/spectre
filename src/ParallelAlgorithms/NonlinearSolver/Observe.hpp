// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/PrettyType.hpp"

namespace NonlinearSolver::observe_detail {

using reduction_data = Parallel::ReductionData<
    // Iteration
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Globalization step
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Residual
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    // Step length
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
 */
template <typename OptionsGroup, typename Metavariables>
void contribute_to_reduction_observer(
    const size_t iteration_id, const size_t globalization_iteration_id,
    const double residual_magnitude, const double step_length,
    Parallel::GlobalCache<Metavariables>& cache) noexcept {
  const auto observation_id = observers::ObservationId(
      iteration_id, pretty_type::get_name<OptionsGroup>());
  auto& reduction_writer = Parallel::get_parallel_component<
      observers::ObserverWriter<Metavariables>>(cache);
  Parallel::threaded_action<observers::ThreadedActions::WriteReductionData>(
      // Node 0 is always the writer, so directly call the component on that
      // node
      reduction_writer[0], observation_id,
      static_cast<size_t>(Parallel::my_node()),
      std::string{"/" + Options::name<OptionsGroup>() + "Residuals"},
      std::vector<std::string>{"Iteration", "GlobalizationStep", "Residual",
                               "StepLength"},
      reduction_data{iteration_id, globalization_iteration_id,
                     residual_magnitude, step_length});
}

}  // namespace NonlinearSolver::observe_detail
