// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace NonlinearSolver::observe_detail {

/*!
 * \brief Contributes data from the residual monitor to the reduction observer
 */
template <typename OptionsGroup, typename ParallelComponent,
          typename Metavariables>
void contribute_to_reduction_observer(
    const size_t iteration_id, const size_t globalization_iteration_id,
    const double residual_magnitude, const double step_length,
    Parallel::GlobalCache<Metavariables>& cache) {
  auto& reduction_writer = Parallel::get_parallel_component<
      observers::ObserverWriter<Metavariables>>(cache);
  Parallel::threaded_action<observers::ThreadedActions::WriteReductionDataRow>(
      // Node 0 is always the writer, so directly call the component on that
      // node
      reduction_writer[0],
      std::string{"/" + pretty_type::name<OptionsGroup>() + "Residuals"},
      std::vector<std::string>{"Iteration", "Walltime", "GlobalizationStep",
                               "Residual", "StepLength"},
      std::make_tuple(iteration_id, sys::wall_time(),
                      globalization_iteration_id, residual_magnitude,
                      step_length));
}

}  // namespace NonlinearSolver::observe_detail
