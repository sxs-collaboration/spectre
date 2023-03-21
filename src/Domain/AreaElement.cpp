// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/AreaElement.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <size_t VolumeDim, typename TargetFrame>
void euclidean_area_element(
    const gsl::not_null<Scalar<DataVector>*> result,
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Scalar<DataVector>& inverse_jacobian_determinant_face,
    const Direction<VolumeDim>& direction) {
  ASSERT(inverse_jacobian_face.get(0, 0).size() ==
             get(inverse_jacobian_determinant_face).size(),
         "InverseJacobian and determinant are expected to have the same number "
         "of grid points but have "
             << inverse_jacobian_face.get(0, 0).size() << " and "
             << get(inverse_jacobian_determinant_face).size());
  destructive_resize_components(result,
                                get(inverse_jacobian_determinant_face).size());
  get(*result) = square(inverse_jacobian_face.get(direction.dimension(), 0));
  for (size_t d = 1; d < VolumeDim; ++d) {
    get(*result) += square(inverse_jacobian_face.get(direction.dimension(), d));
  }
  get(*result) = sqrt(get(*result)) / get(inverse_jacobian_determinant_face);
}

template <size_t VolumeDim, typename TargetFrame>
void euclidean_area_element(
    const gsl::not_null<Scalar<DataVector>*> result,
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Direction<VolumeDim>& direction) {
  euclidean_area_element(result, inverse_jacobian_face,
                         determinant(inverse_jacobian_face), direction);
}

template <size_t VolumeDim, typename TargetFrame>
Scalar<DataVector> euclidean_area_element(
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Scalar<DataVector>& inverse_jacobian_determinant_face,
    const Direction<VolumeDim>& direction) {
  Scalar<DataVector> result{};
  euclidean_area_element(make_not_null(&result), inverse_jacobian_face,
                         inverse_jacobian_determinant_face, direction);
  return result;
}

template <size_t VolumeDim, typename TargetFrame>
Scalar<DataVector> euclidean_area_element(
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Direction<VolumeDim>& direction) {
  Scalar<DataVector> result{};
  euclidean_area_element(make_not_null(&result), inverse_jacobian_face,
                         determinant(inverse_jacobian_face), direction);
  return result;
}

template <size_t VolumeDim, typename TargetFrame>
void area_element(
    const gsl::not_null<Scalar<DataVector>*> result,
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Scalar<DataVector>& inverse_jacobian_determinant_face,
    const Direction<VolumeDim>& direction,
    const tnsr::II<DataVector, VolumeDim, TargetFrame>& inverse_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  ASSERT(inverse_jacobian_face.get(0, 0).size() ==
             get(inverse_jacobian_determinant_face).size(),
         "InverseJacobian and determinant are expected to have the same number "
         "of grid points but have "
             << inverse_jacobian_face.get(0, 0).size() << " and "
             << get(inverse_jacobian_determinant_face).size());
  result->get() = DataVector(get(inverse_jacobian_determinant_face).size(), 0.);
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = 0; j < VolumeDim; ++j) {
      get(*result) += inverse_spatial_metric.get(i, j) *
                      inverse_jacobian_face.get(direction.dimension(), i) *
                      inverse_jacobian_face.get(direction.dimension(), j);
    }
  }
  get(*result) = sqrt(get(*result)) * get(sqrt_det_spatial_metric) /
                 get(inverse_jacobian_determinant_face);
}

template <size_t VolumeDim, typename TargetFrame>
void area_element(
    const gsl::not_null<Scalar<DataVector>*> result,
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Direction<VolumeDim>& direction,
    const tnsr::II<DataVector, VolumeDim, TargetFrame>& inverse_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  area_element(result, inverse_jacobian_face,
               determinant(inverse_jacobian_face), direction,
               inverse_spatial_metric, sqrt_det_spatial_metric);
}

template <size_t VolumeDim, typename TargetFrame>
Scalar<DataVector> area_element(
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Scalar<DataVector>& inverse_jacobian_determinant_face,
    const Direction<VolumeDim>& direction,
    const tnsr::II<DataVector, VolumeDim, TargetFrame>& inverse_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  Scalar<DataVector> result{};
  area_element(make_not_null(&result), inverse_jacobian_face,
               inverse_jacobian_determinant_face, direction,
               inverse_spatial_metric, sqrt_det_spatial_metric);
  return result;
}

template <size_t VolumeDim, typename TargetFrame>
Scalar<DataVector> area_element(
    const InverseJacobian<DataVector, VolumeDim, Frame::ElementLogical,
                          TargetFrame>& inverse_jacobian_face,
    const Direction<VolumeDim>& direction,
    const tnsr::II<DataVector, VolumeDim, TargetFrame>& inverse_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  Scalar<DataVector> result{};
  area_element(make_not_null(&result), inverse_jacobian_face,
               determinant(inverse_jacobian_face), direction,
               inverse_spatial_metric, sqrt_det_spatial_metric);
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                              \
  template void euclidean_area_element(                                   \
      const gsl::not_null<Scalar<DataVector>*> result,                    \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian_face,          \
      const Scalar<DataVector>& inverse_jacobian_determinant_face,        \
      const Direction<DIM(data)>& direction);                             \
  template void euclidean_area_element(                                   \
      const gsl::not_null<Scalar<DataVector>*> result,                    \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian_face,          \
      const Direction<DIM(data)>& direction);                             \
  template Scalar<DataVector> euclidean_area_element(                     \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian_face,          \
      const Scalar<DataVector>& inverse_jacobian_determinant_face,        \
      const Direction<DIM(data)>& direction);                             \
  template Scalar<DataVector> euclidean_area_element(                     \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian_face,          \
      const Direction<DIM(data)>& direction);                             \
  template void area_element(                                             \
      const gsl::not_null<Scalar<DataVector>*> result,                    \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian_face,          \
      const Scalar<DataVector>& inverse_jacobian_determinant_face,        \
      const Direction<DIM(data)>& direction,                              \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric,                                         \
      const Scalar<DataVector>& sqrt_det_spatial_metric);                 \
  template void area_element(                                             \
      const gsl::not_null<Scalar<DataVector>*> result,                    \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian_face,          \
      const Direction<DIM(data)>& direction,                              \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric,                                         \
      const Scalar<DataVector>& sqrt_det_spatial_metric);                 \
  template Scalar<DataVector> area_element(                               \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian_face,          \
      const Scalar<DataVector>& inverse_jacobian_determinant_face,        \
      const Direction<DIM(data)>& direction,                              \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric,                                         \
      const Scalar<DataVector>& sqrt_det_spatial_metric);                 \
  template Scalar<DataVector> area_element(                               \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical, \
                            FRAME(data)>& inverse_jacobian_face,          \
      const Direction<DIM(data)>& direction,                              \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric,                                         \
      const Scalar<DataVector>& sqrt_det_spatial_metric);
GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3), (Frame::Grid, Frame::Inertial))

#undef DIM
#undef FRAME
#undef INSTANTIATE
