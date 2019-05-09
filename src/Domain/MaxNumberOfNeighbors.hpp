// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \ingroup ComputationalDomainGroup
/// Returns the maximum number of neighbors an element can have in `dim`
/// dimensions.
///
/// \note Assumes a maximum 2-to-1 refinement between two adjacent Elements.
constexpr size_t maximum_number_of_neighbors(const size_t dim) {
  switch (dim) {
    case 1:
      return 2;
    case 2:
      return 8;
    case 3:
      return 24;
    default:
      // need to throw because we cannot ERROR in constexpr
      throw "Invalid dim specified";
  };
}

/// \ingroup ComputationalDomainGroup
/// Returns the maximum number of neighbors in each direction an element can
/// have in `dim` dimensions.
///
/// \note Assumes a maximum 2-to-1 refinement between two adjacent Elements.
constexpr size_t maximum_number_of_neighbors_per_direction(const size_t dim) {
  switch (dim) {
    case 0:
      return 0;
    default:
      return two_to_the(dim - 1);
  };
}
