// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Characteristics.hpp"

#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Assert.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace Burgers {
namespace Tags {
void CharacteristicSpeedsCompute::function(
    const gsl::not_null<return_type*> result,
    const Scalar<DataVector>& u,
    const tnsr::i<DataVector, 1>& normal) noexcept {
  ASSERT(get(u).size() == 1 and get<0>(normal).size() == 1,
         "Char speeds only written for 1d boundaries.  "
         "Got wrong number of points: "
         << get(u).size() << " and " << get<0>(normal).size());
  gsl::at(*result, 0) = get<0>(normal)[0] > 0.0 ? get(u) : -get(u);
}
}  // namespace Tags
}  // namespace Burgers
/// \endcond
