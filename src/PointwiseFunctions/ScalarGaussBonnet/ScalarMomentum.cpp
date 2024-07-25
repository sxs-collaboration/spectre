// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/ScalarGaussBonnet/ScalarMomentum.hpp"

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "Elliptic/Systems/ScalarGaussBonnet/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace sgb {

void scalar_momentum(const gsl::not_null<Scalar<DataVector>*> result,
                     const tnsr::i<DataVector, 3, Frame::Inertial>& deriv,
                     const tnsr::I<DataVector, 3>& shift,
                     const Scalar<DataVector>& lapse) {
  dot_product(result, shift, deriv);
  get(*result) /= get(lapse);
}

Scalar<DataVector> scalar_momentum(
    const tnsr::i<DataVector, 3, Frame::Inertial>& deriv,
    const tnsr::I<DataVector, 3>& shift, const Scalar<DataVector>& lapse) {
  Scalar<DataVector> result{get(lapse).size()};
  scalar_momentum(make_not_null(&result), deriv, shift, lapse);
  return result;
}

}  // namespace sgb

template struct sgb::Tags::PiCompute<gr::Tags::Shift<DataVector, 3>,
                                     CurvedScalarWave::Tags::Pi>;

template struct sgb::Tags::PiCompute<sgb::Tags::RolledOffShift,
                                     sgb::Tags::PiWithRolledOffShift>;
