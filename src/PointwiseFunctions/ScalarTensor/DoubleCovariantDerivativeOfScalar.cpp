// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/ScalarTensor/DoubleCovariantDerivativeOfScalar.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarTensor {

template <typename DataType, typename Frame>
void DDKG_normal_normal_projection(
    gsl::not_null<Scalar<DataType>*> DDKG_normal_normal_result,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, 3, Frame>& shift,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_pi_scalar,
    const Scalar<DataType>& dt_pi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_lapse) {
  tenex::evaluate(
      DDKG_normal_normal_result,
      // - L_n Pi - (1/lapse) Phi^{i} partial_i lapse
      -(1.0 / lapse()) * (dt_pi_scalar() - shift(ti::I) * d_pi_scalar(ti::i)

                          + inverse_spatial_metric(ti::I, ti::J) *
                                phi_scalar(ti::i) * d_lapse(ti::j)

                              )

  );
}

template <typename DataType, typename Frame>
Scalar<DataType> DDKG_normal_normal_projection(
    const Scalar<DataType>& lapse, const tnsr::I<DataType, 3, Frame>& shift,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_pi_scalar,
    const Scalar<DataType>& dt_pi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_lapse) {
  Scalar<DataType> result{};
  DDKG_normal_normal_projection(make_not_null(&result), lapse, shift,
                                inverse_spatial_metric, phi_scalar, d_pi_scalar,
                                dt_pi_scalar, d_lapse);
  return result;
}

template <typename DataType, typename Frame>
void DDKG_normal_spatial_projection(
    gsl::not_null<tnsr::i<DataType, 3, Frame>*> DDKG_normal_spatial_result,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, 3, Frame>& extrinsic_curvature,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_pi_scalar) {
  tenex::evaluate<ti::i>(
      DDKG_normal_spatial_result, extrinsic_curvature(ti::i, ti::j) *
                                          inverse_spatial_metric(ti::J, ti::K) *
                                          phi_scalar(ti::k)

                                      - d_pi_scalar(ti::i)

  );
}

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame> DDKG_normal_spatial_projection(
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, 3, Frame>& extrinsic_curvature,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_pi_scalar) {
  tnsr::i<DataType, 3, Frame> result{};
  DDKG_normal_spatial_projection(make_not_null(&result), inverse_spatial_metric,
                                 extrinsic_curvature, phi_scalar, d_pi_scalar);
  return result;
}

template <typename DataType, typename Frame>
void DDKG_spatial_spatial_projection(
    gsl::not_null<tnsr::ii<DataType, 3, Frame>*> DDKG_spatial_spatial_result,
    const tnsr::ii<DataType, 3, Frame>& extrinsic_curvature,
    const tnsr::Ijj<DataType, 3, Frame>& spatial_christoffel_second_kind,
    const Scalar<DataType>& pi_scalar,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::ij<DataType, 3, Frame>& d_phi_scalar) {
  // Note that D_phi is the covariant derivative and has Christoffel symbols
  tenex::evaluate<ti::i, ti::j>(
      DDKG_spatial_spatial_result,
      -pi_scalar() * extrinsic_curvature(ti::i, ti::j)
          // Note covariant derivative
          // and symmetrize partial derivative of scalar
          + 0.5 * (d_phi_scalar(ti::i, ti::j) + d_phi_scalar(ti::j, ti::i)) -
          spatial_christoffel_second_kind(ti::K, ti::i, ti::j) *
              phi_scalar(ti::k));
}

template <typename DataType, typename Frame>
tnsr::ii<DataType, 3, Frame> DDKG_spatial_spatial_projection(
    const tnsr::ii<DataType, 3, Frame>& extrinsic_curvature,
    const tnsr::Ijj<DataType, 3, Frame>& spatial_christoffel_second_kind,
    const Scalar<DataType>& pi_scalar,
    const tnsr::i<DataType, 3, Frame>& phi_scalar,
    const tnsr::ij<DataType, 3, Frame>& d_phi_scalar) {
  tnsr::ii<DataType, 3, Frame> result{};
  DDKG_spatial_spatial_projection(make_not_null(&result), extrinsic_curvature,
                                  spatial_christoffel_second_kind, pi_scalar,
                                  phi_scalar, d_phi_scalar);
  return result;
}
}  // namespace ScalarTensor

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template void ScalarTensor::DDKG_normal_normal_projection(                \
      gsl::not_null<Scalar<DTYPE(data)>*> DDKG_normal_normal_result,        \
      const Scalar<DTYPE(data)>& lapse,                                     \
      const tnsr::I<DTYPE(data), 3, FRAME(data)>& shift,                    \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>& inverse_spatial_metric,  \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& phi_scalar,               \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& d_pi_scalar,              \
      const Scalar<DTYPE(data)>& dt_pi_scalar,                              \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& d_lapse);                 \
  template Scalar<DTYPE(data)> ScalarTensor::DDKG_normal_normal_projection( \
      const Scalar<DTYPE(data)>& lapse,                                     \
      const tnsr::I<DTYPE(data), 3, FRAME(data)>& shift,                    \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>& inverse_spatial_metric,  \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& phi_scalar,               \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& d_pi_scalar,              \
      const Scalar<DTYPE(data)>& dt_pi_scalar,                              \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& d_lapse);                 \
  template void ScalarTensor::DDKG_normal_spatial_projection(               \
      gsl::not_null<tnsr::i<DTYPE(data), 3, FRAME(data)>*>                  \
          DDKG_normal_spatial_result,                                       \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>& inverse_spatial_metric,  \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& extrinsic_curvature,     \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& phi_scalar,               \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& d_pi_scalar);             \
  template tnsr::i<DTYPE(data), 3, FRAME(data)>                             \
  ScalarTensor::DDKG_normal_spatial_projection(                             \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>& inverse_spatial_metric,  \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& extrinsic_curvature,     \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& phi_scalar,               \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& d_pi_scalar);             \
  template void ScalarTensor::DDKG_spatial_spatial_projection(              \
      gsl::not_null<tnsr::ii<DTYPE(data), 3, FRAME(data)>*>                 \
          DDKG_spatial_spatial_result,                                      \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& extrinsic_curvature,     \
      const tnsr::Ijj<DTYPE(data), 3, FRAME(data)>&                         \
          spatial_christoffel_second_kind,                                  \
      const Scalar<DTYPE(data)>& pi_scalar,                                 \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& phi_scalar,               \
      const tnsr::ij<DTYPE(data), 3, FRAME(data)>& d_phi_scalar);           \
  template tnsr::ii<DTYPE(data), 3, FRAME(data)>                            \
  ScalarTensor::DDKG_spatial_spatial_projection(                            \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& extrinsic_curvature,     \
      const tnsr::Ijj<DTYPE(data), 3, FRAME(data)>&                         \
          spatial_christoffel_second_kind,                                  \
      const Scalar<DTYPE(data)>& pi_scalar,                                 \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& phi_scalar,               \
      const tnsr::ij<DTYPE(data), 3, FRAME(data)>& d_phi_scalar);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DTYPE
#undef FRAME
#undef INSTANTIATE
