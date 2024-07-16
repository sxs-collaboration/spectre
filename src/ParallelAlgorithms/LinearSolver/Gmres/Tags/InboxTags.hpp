// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <tuple>

#include "DataStructures/DynamicVector.hpp"
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

template <typename OptionsGroup, typename ValueType>
struct Orthogonalization : Parallel::InboxInserters::Value<
                               Orthogonalization<OptionsGroup, ValueType>> {
  using temporal_id = size_t;
  using type = std::map<temporal_id, ValueType>;
};

template <typename OptionsGroup, typename ValueType>
struct FinalOrthogonalization
    : Parallel::InboxInserters::Value<
          FinalOrthogonalization<OptionsGroup, ValueType>> {
  using temporal_id = size_t;
  using type =
      std::map<temporal_id, std::tuple<double, blaze::DynamicVector<ValueType>,
                                       Convergence::HasConverged>>;
};

}  // namespace LinearSolver::gmres::detail::Tags
