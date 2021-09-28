// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/TimeDerivative.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarWave {
template <size_t Dim>
void TimeDerivative<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,
    const gsl::not_null<Scalar<DataVector>*> dt_psi,

    const gsl::not_null<Scalar<DataVector>*> result_gamma2,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_pi,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const Scalar<DataVector>& gamma2) {
  // The constraint damping parameter gamma2 is needed for boundary corrections,
  // which means we need it as a temporary tag in order to project it to the
  // boundary. We prevent slicing/projecting directly from the volume to prevent
  // people from adding many compute tags to the DataBox, instead preferring
  // quantities be computed inside the TimeDerivative/Flux/Source structs. This
  // keeps related code together and makes figuring out where something is
  // computed a lot easier.
  *result_gamma2 = gamma2;

  get(*dt_psi) = -get(pi);
  get(*dt_pi) = -get<0, 0>(d_phi);
  for (size_t d = 1; d < Dim; ++d) {
    get(*dt_pi) -= d_phi.get(d, d);
  }
  for (size_t d = 0; d < Dim; ++d) {
    dt_phi->get(d) = -d_pi.get(d) + get(gamma2) * (d_psi.get(d) - phi.get(d));
  }
}

template class TimeDerivative<1>;
template class TimeDerivative<2>;
template class TimeDerivative<3>;
}  // namespace ScalarWave
