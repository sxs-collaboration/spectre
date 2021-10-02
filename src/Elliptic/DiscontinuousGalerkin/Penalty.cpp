// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/DiscontinuousGalerkin/Penalty.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace elliptic::dg {

DataVector penalty(const DataVector& element_size, const size_t num_points,
                   const double penalty_parameter) {
  return penalty_parameter * square(num_points) / element_size;
}

}  // namespace elliptic::dg
