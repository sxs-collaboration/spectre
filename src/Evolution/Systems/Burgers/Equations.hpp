// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
// IWYU pragma: no_forward_declare Tensor
namespace Burgers {
namespace Tags {
struct U;
}  // namespace Tags
}  // namespace Burgers
/// \endcond

namespace Burgers {
struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<Tags::U>;
  static double apply(const Scalar<DataVector>& u) noexcept;
};
}  // namespace Burgers
