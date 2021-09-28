// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace gr {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> weyl_propagating(
    const tnsr::ii<DataType, SpatialDim, Frame>& ricci,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::I<DataType, SpatialDim, Frame>& unit_interface_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& projection_IJ,
    const tnsr::ii<DataType, SpatialDim, Frame>& projection_ij,
    const tnsr::Ij<DataType, SpatialDim, Frame>& projection_Ij,
    const double sign) {
  tnsr::ii<DataType, SpatialDim, Frame> weyl_prop(
      get_size(get<0>(unit_interface_normal_vector)));
  weyl_propagating<SpatialDim, Frame, DataType>(
      make_not_null(&weyl_prop), ricci, extrinsic_curvature,
      inverse_spatial_metric, cov_deriv_extrinsic_curvature,
      unit_interface_normal_vector, projection_IJ, projection_ij, projection_Ij,
      sign);
  return weyl_prop;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_propagating(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> weyl_prop_u8,
    const tnsr::ii<DataType, SpatialDim, Frame>& ricci,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::I<DataType, SpatialDim, Frame>& unit_interface_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& projection_IJ,
    const tnsr::ii<DataType, SpatialDim, Frame>& projection_ij,
    const tnsr::Ij<DataType, SpatialDim, Frame>& projection_Ij,
    const double sign) {
  ASSERT((sign == 1.) or (sign == -1.),
         "Calculation of weyl propagating modes accepts only +1/-1 to indicate "
         "which of U8+/- is needed.");
  destructive_resize_components(weyl_prop_u8,
                                get_size(get<0>(unit_interface_normal_vector)));

  TempBuffer<tmpl::list<::Tags::Tempii<0, SpatialDim, Frame, DataType>>>
      unprojected_weyl_prop_u8_vars(
          get_size(get<0>(unit_interface_normal_vector)));

  auto& unprojected_weyl_prop_u8 =
      get<::Tags::Tempii<0, SpatialDim, Frame, DataType>>(
          unprojected_weyl_prop_u8_vars);

  gr::weyl_electric(make_not_null(&unprojected_weyl_prop_u8), ricci,
                    extrinsic_curvature, inverse_spatial_metric);

  // Compute the portion that the projections act on
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {  // Symmetry
      for (size_t k = 0; k < SpatialDim; ++k) {
        unprojected_weyl_prop_u8.get(i, j) -=
            sign * unit_interface_normal_vector.get(k) *
            (cov_deriv_extrinsic_curvature.get(k, i, j) -
             0.5 * (cov_deriv_extrinsic_curvature.get(j, i, k) +
                    cov_deriv_extrinsic_curvature.get(i, j, k)));
      }
    }
  }

  // Now project
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {  // Symmetry
      weyl_prop_u8->get(i, j) = 0.;
      for (size_t k = 0; k < SpatialDim; ++k) {
        for (size_t l = 0; l < SpatialDim; ++l) {
          weyl_prop_u8->get(i, j) +=
              (projection_Ij.get(k, i) * projection_Ij.get(l, j) -
               0.5 * projection_IJ.get(k, l) * projection_ij.get(i, j)) *
              unprojected_weyl_prop_u8.get(k, l);
        }
      }
    }
  }
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                   \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)> gr::weyl_propagating( \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& ricci,              \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                     \
          extrinsic_curvature,                                                 \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spatial_metric,                                              \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                    \
          cov_deriv_extrinsic_curvature,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>&                      \
          unit_interface_normal_vector,                                        \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>& projection_IJ,      \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& projection_ij,      \
      const tnsr::Ij<DTYPE(data), DIM(data), FRAME(data)>& projection_Ij,      \
      const double sign);                                                      \
  template void gr::weyl_propagating(                                          \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*>      \
          weyl_prop_u8,                                                        \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& ricci,              \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                     \
          extrinsic_curvature,                                                 \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spatial_metric,                                              \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                    \
          cov_deriv_extrinsic_curvature,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>&                      \
          unit_interface_normal_vector,                                        \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>& projection_IJ,      \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& projection_ij,      \
      const tnsr::Ij<DTYPE(data), DIM(data), FRAME(data)>& projection_Ij,      \
      const double sign);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
