// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/RicciScalar.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr::surfaces {
template <typename Frame>
void ricci_scalar(const gsl::not_null<Scalar<DataVector>*> result,
                  const tnsr::ii<DataVector, 3, Frame>& spatial_ricci_tensor,
                  const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
                  const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
                  const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric) {
  trace(result, spatial_ricci_tensor, upper_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(*result) -= 2.0 * spatial_ricci_tensor.get(i, j) *
                      unit_normal_vector.get(i) * unit_normal_vector.get(j);

      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          // K^{ij} K_{ij} = g^{ik} g^{jl} K_{kl} K_{ij}
          get(*result) -=
              upper_spatial_metric.get(i, k) * upper_spatial_metric.get(j, l) *
              extrinsic_curvature.get(k, l) * extrinsic_curvature.get(i, j);
        }
      }
    }
  }
  get(*result) += square(get(trace(extrinsic_curvature, upper_spatial_metric)));
}

template <typename Frame>
Scalar<DataVector> ricci_scalar(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci_tensor,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric) {
  Scalar<DataVector> result{};
  ricci_scalar(make_not_null(&result), spatial_ricci_tensor, unit_normal_vector,
               extrinsic_curvature, upper_spatial_metric);
  return result;
}
}  // namespace gr::surfaces

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                             \
  template void gr::surfaces::ricci_scalar<FRAME(data)>(                 \
      const gsl::not_null<Scalar<DataVector>*> result,                   \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_ricci_tensor,  \
      const tnsr::I<DataVector, 3, FRAME(data)>& unit_normal_vector,     \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature,   \
      const tnsr::II<DataVector, 3, FRAME(data)>& upper_spatial_metric); \
  template Scalar<DataVector> gr::surfaces::ricci_scalar<FRAME(data)>(   \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_ricci_tensor,  \
      const tnsr::I<DataVector, 3, FRAME(data)>& unit_normal_vector,     \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature,   \
      const tnsr::II<DataVector, 3, FRAME(data)>& upper_spatial_metric);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
