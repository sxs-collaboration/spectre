// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Fluxes.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace Burgers {
void Fluxes::apply(const gsl::not_null<tnsr::I<DataVector, 1>*> flux,
                   const Scalar<DataVector>& u) noexcept {
  get<0>(*flux) = 0.5 * square(get(u));
}
}  // namespace Burgers
