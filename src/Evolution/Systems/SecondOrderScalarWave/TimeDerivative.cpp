// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SecondOrderScalarWave/TimeDerivative.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace SecondOrderScalarWave {
template <size_t Dim>
evolution::dg::TimeDerivativeDecisions<Dim> TimeDerivative<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*d_psi*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*d_pi*/,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi,
    const Scalar<DataVector>& pi) {
  get(*dt_psi) = -get(pi);
  get(*dt_pi) = 0.0;
  for (size_t d = 0; d < Dim; ++d) {
    get(*dt_pi) -= d_phi.get(d, d);
  }
  return {false};
}

template class TimeDerivative<1>;
template class TimeDerivative<2>;
template class TimeDerivative<3>;
}  // namespace SecondOrderScalarWave
