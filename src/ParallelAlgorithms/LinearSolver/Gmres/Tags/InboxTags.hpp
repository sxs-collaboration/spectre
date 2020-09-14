// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <tuple>

#include "DataStructures/DenseVector.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::gmres::detail::Tags {

template <typename OptionsGroup>
struct InitialOrthogonalization
    : Parallel::InboxInserters::Value<InitialOrthogonalization<OptionsGroup>> {
  using temporal_id = size_t;
  using type =
      std::map<temporal_id, std::tuple<double, Convergence::HasConverged>>;
};

template <typename OptionsGroup>
struct Orthogonalization
    : Parallel::InboxInserters::Value<Orthogonalization<OptionsGroup>> {
  using temporal_id = size_t;
  using type = std::map<temporal_id, double>;
};

template <typename OptionsGroup>
struct FinalOrthogonalization
    : Parallel::InboxInserters::Value<FinalOrthogonalization<OptionsGroup>> {
  using temporal_id = size_t;
  using type = std::map<temporal_id, std::tuple<double, DenseVector<double>,
                                                Convergence::HasConverged>>;
};

}  // namespace LinearSolver::gmres::detail::Tags
