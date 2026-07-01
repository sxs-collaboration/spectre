// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/DoubleCovariantDerivativeOfCoupling.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace ScalarTensor::sgb {

template <typename DataType>
void DDCoupling_normal_normal_projection(
    const gsl::not_null<Scalar<DataType>*> DDCoupling_normal_normal_result,
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const Scalar<DataType>& pi_scalar,
    const Scalar<DataType>& normal_normal_DD_scalar) {
  get(*DDCoupling_normal_normal_result) =
      get(coupling_prime_prime) * square(get(pi_scalar)) +
      get(coupling_prime) * get(normal_normal_DD_scalar);
}

template <typename DataType>
Scalar<DataType> DDCoupling_normal_normal_projection(
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const Scalar<DataType>& pi_scalar,
    const Scalar<DataType>& normal_normal_DD_scalar) {
  Scalar<DataType> result{};
  DDCoupling_normal_normal_projection(make_not_null(&result), coupling_prime,
                                      coupling_prime_prime, pi_scalar,
                                      normal_normal_DD_scalar);
  return result;
}

template <typename DataType, typename Frame>
void DDCoupling_normal_spatial_projection(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*>
        DDCoupling_normal_spatial_result,
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const Scalar<DataType>& pi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_scalar_field,
    const tnsr::i<DataType, 3, Frame>& normal_spatial_DD_scalar) {
  tenex::evaluate<ti::i>(
      DDCoupling_normal_spatial_result,
      -coupling_prime_prime() * pi_scalar() * d_scalar_field(ti::i) +
          coupling_prime() * normal_spatial_DD_scalar(ti::i));
}

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame> DDCoupling_normal_spatial_projection(
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const Scalar<DataType>& pi_scalar,
    const tnsr::i<DataType, 3, Frame>& d_scalar_field,
    const tnsr::i<DataType, 3, Frame>& normal_spatial_DD_scalar) {
  tnsr::i<DataType, 3, Frame> result{};
  DDCoupling_normal_spatial_projection(
      make_not_null(&result), coupling_prime, coupling_prime_prime, pi_scalar,
      d_scalar_field, normal_spatial_DD_scalar);
  return result;
}

template <typename DataType, typename Frame>
void DDCoupling_spatial_spatial_projection(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*>
        DDCoupling_spatial_spatial_result,
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const tnsr::i<DataType, 3, Frame>& d_scalar_field,
    const tnsr::ii<DataType, 3, Frame>& spatial_spatial_DD_scalar) {
  tenex::evaluate<ti::i, ti::j>(
      DDCoupling_spatial_spatial_result,
      coupling_prime_prime() * d_scalar_field(ti::i) * d_scalar_field(ti::j) +
          coupling_prime() * spatial_spatial_DD_scalar(ti::i, ti::j));
}

template <typename DataType, typename Frame>
tnsr::ii<DataType, 3, Frame> DDCoupling_spatial_spatial_projection(
    const Scalar<DataType>& coupling_prime,
    const Scalar<DataType>& coupling_prime_prime,
    const tnsr::i<DataType, 3, Frame>& d_scalar_field,
    const tnsr::ii<DataType, 3, Frame>& spatial_spatial_DD_scalar) {
  tnsr::ii<DataType, 3, Frame> result{};
  DDCoupling_spatial_spatial_projection(make_not_null(&result), coupling_prime,
                                        coupling_prime_prime, d_scalar_field,
                                        spatial_spatial_DD_scalar);
  return result;
}
}  // namespace ScalarTensor::sgb

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALAR_FNS(_, data)                                    \
  template void ScalarTensor::sgb::DDCoupling_normal_normal_projection(    \
      gsl::not_null<Scalar<DTYPE(data)>*> DDCoupling_normal_normal_result, \
      const Scalar<DTYPE(data)>& coupling_prime,                           \
      const Scalar<DTYPE(data)>& coupling_prime_prime,                     \
      const Scalar<DTYPE(data)>& pi_scalar,                                \
      const Scalar<DTYPE(data)>& normal_normal_DD_scalar);                 \
  template Scalar<DTYPE(data)>                                             \
  ScalarTensor::sgb::DDCoupling_normal_normal_projection(                  \
      const Scalar<DTYPE(data)>& coupling_prime,                           \
      const Scalar<DTYPE(data)>& coupling_prime_prime,                     \
      const Scalar<DTYPE(data)>& pi_scalar,                                \
      const Scalar<DTYPE(data)>& normal_normal_DD_scalar);

#define INSTANTIATE_TENSOR_FNS(_, data)                                        \
  template void ScalarTensor::sgb::DDCoupling_normal_spatial_projection(       \
      gsl::not_null<tnsr::i<DTYPE(data), 3, FRAME(data)>*>                     \
          DDCoupling_normal_spatial_result,                                    \
      const Scalar<DTYPE(data)>& coupling_prime,                               \
      const Scalar<DTYPE(data)>& coupling_prime_prime,                         \
      const Scalar<DTYPE(data)>& pi_scalar,                                    \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& d_scalar_field,              \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& normal_spatial_DD_scalar);   \
  template tnsr::i<DTYPE(data), 3, FRAME(data)>                                \
  ScalarTensor::sgb::DDCoupling_normal_spatial_projection(                     \
      const Scalar<DTYPE(data)>& coupling_prime,                               \
      const Scalar<DTYPE(data)>& coupling_prime_prime,                         \
      const Scalar<DTYPE(data)>& pi_scalar,                                    \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& d_scalar_field,              \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& normal_spatial_DD_scalar);   \
  template void ScalarTensor::sgb::DDCoupling_spatial_spatial_projection(      \
      gsl::not_null<tnsr::ii<DTYPE(data), 3, FRAME(data)>*>                    \
          DDCoupling_spatial_spatial_result,                                   \
      const Scalar<DTYPE(data)>& coupling_prime,                               \
      const Scalar<DTYPE(data)>& coupling_prime_prime,                         \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& d_scalar_field,              \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& spatial_spatial_DD_scalar); \
  template tnsr::ii<DTYPE(data), 3, FRAME(data)>                               \
  ScalarTensor::sgb::DDCoupling_spatial_spatial_projection(                    \
      const Scalar<DTYPE(data)>& coupling_prime,                               \
      const Scalar<DTYPE(data)>& coupling_prime_prime,                         \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& d_scalar_field,              \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& spatial_spatial_DD_scalar);

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALAR_FNS, (double, DataVector))
GENERATE_INSTANTIATIONS(INSTANTIATE_TENSOR_FNS, (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DTYPE
#undef FRAME
#undef INSTANTIATE_SCALAR_FNS
#undef INSTANTIATE_TENSOR_FNS
