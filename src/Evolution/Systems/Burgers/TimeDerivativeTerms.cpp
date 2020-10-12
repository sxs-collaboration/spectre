// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/TimeDerivativeTerms.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Burgers {
void TimeDerivativeTerms::apply(
    const gsl::not_null<Scalar<DataVector>*> /*non_flux_terms_dt_vars*/,
    const gsl::not_null<tnsr::I<DataVector, 1, Frame::Inertial>*> flux_u,
    const Scalar<DataVector>& u) noexcept {
  get<0>(*flux_u) = 0.5 * square(get(u));
}
}  // namespace Burgers
/// \endcond
