// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/TimeDerivative.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Burgers {
void TimeDerivative::apply(
    const gsl::not_null<Scalar<DataVector>*> /*dt_vars*/,
    const gsl::not_null<tnsr::I<DataVector, 1, Frame::Inertial>*> flux_u,
    const Scalar<DataVector>& u) noexcept {
  get<0>(*flux_u) = 0.5 * square(get(u));
}
}  // namespace Burgers
/// \endcond
