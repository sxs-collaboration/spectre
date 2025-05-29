// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/QuadraticCurvatureScalars.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {

template <typename Frame>
void pontryagin_scalar_in_vacuum(
    const gsl::not_null<Scalar<DataVector>*> pontryagin_scalar,
    const tnsr::ii<DataVector, 3, Frame>& weyl_electric,
    const tnsr::ii<DataVector, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric) {
  tenex::evaluate(pontryagin_scalar, -16.0 * weyl_electric(ti::i, ti::j) *
                                         inverse_spatial_metric(ti::J, ti::K) *
                                         weyl_magnetic(ti::k, ti::l) *
                                         inverse_spatial_metric(ti::L, ti::I));
}

template <typename Frame>
Scalar<DataVector> pontryagin_scalar_in_vacuum(
    const tnsr::ii<DataVector, 3, Frame>& weyl_electric,
    const tnsr::ii<DataVector, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric) {
  Scalar<DataVector> pontryagin_scalar{get<0, 0>(weyl_electric).size()};
  pontryagin_scalar_in_vacuum(make_not_null(&pontryagin_scalar), weyl_electric,
                              weyl_magnetic, inverse_spatial_metric);
  return pontryagin_scalar;
}

void gauss_bonnet_scalar_in_vacuum(
    const gsl::not_null<Scalar<DataVector>*> gb_scalar,
    const Scalar<DataVector>& weyl_electric_scalar,
    const Scalar<DataVector>& weyl_magnetic_scalar) {
  // Compute the Kretschmann scalar in vacuum
  gb_scalar->get() =
      8.0 * (weyl_electric_scalar.get() - weyl_magnetic_scalar.get());
}

Scalar<DataVector> gauss_bonnet_scalar_in_vacuum(
    const Scalar<DataVector>& weyl_electric_scalar,
    const Scalar<DataVector>& weyl_magnetic_scalar) {
  Scalar<DataVector> gb_scalar{get(weyl_electric_scalar).size()};
  gauss_bonnet_scalar_in_vacuum(make_not_null(&gb_scalar), weyl_electric_scalar,
                                weyl_magnetic_scalar);
  return gb_scalar;
}

}  // namespace gr

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template Scalar<DataVector> gr::pontryagin_scalar_in_vacuum(             \
      const tnsr::ii<DataVector, 3, FRAME(data)>& weyl_electric,           \
      const tnsr::ii<DataVector, 3, FRAME(data)>& weyl_magnetic,           \
      const tnsr::II<DataVector, 3, FRAME(data)>& inverse_spatial_metric); \
  template void gr::pontryagin_scalar_in_vacuum(                            \
      const gsl::not_null<Scalar<DataVector>*> pontryagin_scalar,          \
      const tnsr::ii<DataVector, 3, FRAME(data)>& weyl_electric,           \
      const tnsr::ii<DataVector, 3, FRAME(data)>& weyl_magnetic,           \
      const tnsr::II<DataVector, 3, FRAME(data)>& inverse_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial))

#undef FRAME
#undef INSTANTIATE
