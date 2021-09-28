// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/CovariantDerivOfExtrinsicCurvature.hpp"

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ijj<DataType, SpatialDim, Frame> covariant_deriv_of_extrinsic_curvature(
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal_vector,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        spatial_christoffel_second_kind,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) {
  tnsr::ijj<DataType, SpatialDim, Frame> d_extrinsic_curvature(
      get_size(get<0>(spacetime_unit_normal_vector)));
  GeneralizedHarmonic::covariant_deriv_of_extrinsic_curvature<SpatialDim, Frame,
                                                              DataType>(
      make_not_null(&d_extrinsic_curvature), extrinsic_curvature,
      spacetime_unit_normal_vector, spatial_christoffel_second_kind,
      inverse_spacetime_metric, phi, d_pi, d_phi);
  return d_extrinsic_curvature;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void covariant_deriv_of_extrinsic_curvature(
    const gsl::not_null<tnsr::ijj<DataType, SpatialDim, Frame>*>
        d_extrinsic_curvature,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal_vector,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        spatial_christoffel_second_kind,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) {
  destructive_resize_components(d_extrinsic_curvature,
                                get_size(get<0>(spacetime_unit_normal_vector)));

  // Ordinary derivative first
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        d_extrinsic_curvature->get(k, i, j) = d_pi.get(k, i + 1, j + 1);
        for (size_t a = 0; a <= SpatialDim; ++a) {
          d_extrinsic_curvature->get(k, i, j) +=
              (d_phi.get(k, i, j + 1, a) + d_phi.get(k, j, i + 1, a)) *
              spacetime_unit_normal_vector.get(a);
          for (size_t b = 0; b <= SpatialDim; ++b) {
            for (size_t c = 0; c <= SpatialDim; ++c) {
              d_extrinsic_curvature->get(k, i, j) -=
                  (phi.get(i, j + 1, a) + phi.get(j, i + 1, a)) *
                  spacetime_unit_normal_vector.get(b) *
                  (inverse_spacetime_metric.get(c, a) +
                   0.5 * spacetime_unit_normal_vector.get(c) *
                       spacetime_unit_normal_vector.get(a)) *
                  phi.get(k, c, b);
            }
          }
        }
      }
    }
  }

  // Now add gamma terms
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        d_extrinsic_curvature->get(k, i, j) =
            0.5 * d_extrinsic_curvature->get(k, i, j) -
            spatial_christoffel_second_kind.get(0, i, k) *
                extrinsic_curvature.get(0, j) -
            spatial_christoffel_second_kind.get(0, j, k) *
                extrinsic_curvature.get(0, i);
        for (size_t l = 1; l < SpatialDim; ++l) {
          d_extrinsic_curvature->get(k, i, j) -=
              spatial_christoffel_second_kind.get(l, i, k) *
                  extrinsic_curvature.get(l, j) +
              spatial_christoffel_second_kind.get(l, j, k) *
                  extrinsic_curvature.get(l, i);
        }
      }
    }
  }
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>                    \
  GeneralizedHarmonic::covariant_deriv_of_extrinsic_curvature(               \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                   \
          extrinsic_curvature,                                               \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_unit_normal_vector,                                      \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spatial_christoffel_second_kind,                                   \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,            \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi);         \
  template void GeneralizedHarmonic::covariant_deriv_of_extrinsic_curvature( \
      gsl::not_null<tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>*>         \
          d_extrinsic_curvature,                                             \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                   \
          extrinsic_curvature,                                               \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_unit_normal_vector,                                      \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spatial_christoffel_second_kind,                                   \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,            \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
