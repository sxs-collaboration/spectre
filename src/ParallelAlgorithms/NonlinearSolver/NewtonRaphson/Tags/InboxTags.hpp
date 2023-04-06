// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <variant>

#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

namespace NonlinearSolver::newton_raphson::detail::Tags {

template <typename OptionsGroup>
struct GlobalizationResult
    : Parallel::InboxInserters::Value<GlobalizationResult<OptionsGroup>> {
  using temporal_id = size_t;
  using type =
      std::map<temporal_id, std::variant<double, Convergence::HasConverged>>;
};

}  // namespace NonlinearSolver::newton_raphson::detail::Tags
