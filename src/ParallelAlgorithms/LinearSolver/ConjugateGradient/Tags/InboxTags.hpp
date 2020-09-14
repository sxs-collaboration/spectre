// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <tuple>

#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::cg::detail::Tags {

template <typename OptionsGroup>
struct InitialHasConverged
    : Parallel::InboxInserters::Value<InitialHasConverged<OptionsGroup>> {
  using temporal_id = size_t;
  using type = std::map<temporal_id, Convergence::HasConverged>;
};

template <typename OptionsGroup>
struct Alpha : Parallel::InboxInserters::Value<Alpha<OptionsGroup>> {
  using temporal_id = size_t;
  using type = std::map<temporal_id, double>;
};

template <typename OptionsGroup>
struct ResidualRatioAndHasConverged
    : Parallel::InboxInserters::Value<
          ResidualRatioAndHasConverged<OptionsGroup>> {
  using temporal_id = size_t;
  using type =
      std::map<temporal_id, std::tuple<double, Convergence::HasConverged>>;
};

}  // namespace LinearSolver::cg::detail::Tags
