// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Characteristics.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Assert.hpp"

/// \cond
namespace Burgers {
namespace Tags {
void CharacteristicSpeedsCompute::function(
    const gsl::not_null<return_type*> result,
    const Scalar<DataVector>& u) noexcept {
  ASSERT(u.size() == 1, "Char speeds only written for 1d boundaries");
  gsl::at(*result, 0) = get(u)[0];
}
}  // namespace Tags
}  // namespace Burgers
/// \endcond
