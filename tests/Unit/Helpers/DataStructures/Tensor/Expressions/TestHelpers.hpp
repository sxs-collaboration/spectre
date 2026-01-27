// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

/// \ingroup TestingFrameworkGroup
/// Functions for testing `TensorExpression`s
namespace TestHelpers::tenex {
// Helper that simply calls `tenex::evaluate`
template <bool ReturnLhsTensor, auto&... LhsTensorIndices, typename LhsTensor,
          typename RhsExpression>
void call_evaluate(const gsl::not_null<LhsTensor*> lhs_tensor,
                   const RhsExpression& rhs_expression) {
  if constexpr (ReturnLhsTensor) {
    *lhs_tensor = ::tenex::evaluate<LhsTensorIndices...>(rhs_expression);
  } else {
    ::tenex::evaluate<LhsTensorIndices...>(lhs_tensor, rhs_expression);
  }
}

// Returns the subset of index positions of an `Index` that a `TensorIndex`
// refers to given the kind of index (e.g. spatial, spacetime, time) that
// `Index` and `TensorIndex` each are
//
// - spatial `Index` and spatial `TensorIndex`: [0, Index::Dim - 1]
// - spacetime `Index` and spacetime `TensorIndex`: [0, Index::Dim - 1]
// - spacetime `Index` and spatial `TensorIndex`: [1, Index::Dim - 1]
// - spacetime `Index` and concrete time `TensorIndex`: [0, 0]
template <typename Index, auto& TensorIndex>
constexpr std::pair<size_t, size_t> get_index_value_range() {
  constexpr bool tensorindex_is_time =
      ::tenex::detail::is_time_index_value(TensorIndex.value);
  static_assert(
      not(Index::index_type == IndexType::Spatial and tensorindex_is_time),
      "Cannot use a concrete time TensorIndex with a SpatialIndex.");
  std::pair<size_t, size_t> range{};
  range.first =
      Index::index_type == IndexType::Spacetime and not TensorIndex.is_spacetime
          ? 1
          : 0;
  range.second = tensorindex_is_time ? 0 : Index::dim - 1;
  return range;
}
}  // namespace TestHelpers::tenex
