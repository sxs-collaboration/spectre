// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/EnergyDensity.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarWave {
template <size_t SpatialDim>
void energy_density(
    gsl::not_null<Scalar<DataVector>*> result, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) noexcept {
  dot_product(result, phi, phi);
  get(*result) += square(get(pi));
  get(*result) *= 0.5;
}

template <size_t SpatialDim>
Scalar<DataVector> energy_density(
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) noexcept {
  Scalar<DataVector> result{get(pi).size()};
  energy_density(make_not_null(&result), pi, phi);
  return result;
}

}  // namespace ScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void ScalarWave::energy_density(                                    \
      gsl::not_null<Scalar<DataVector>*> result, const Scalar<DataVector>& pi, \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
