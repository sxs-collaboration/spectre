// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"

#include <algorithm>

/// Equivalence operator for DataVector
bool operator==(const DataVector& lhs, const DataVector& rhs) noexcept {
  return lhs.size() == rhs.size() and
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

/// Inequivalence operator for DataVector
bool operator!=(const DataVector& lhs, const DataVector& rhs) noexcept {
  return not(lhs == rhs);
}
