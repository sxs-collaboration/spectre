// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/InverseSurfaceMetric.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace StrahlkorperGr {
template <typename Frame>
void inverse_surface_metric(
    const gsl::not_null<tnsr::II<DataVector, 3, Frame>*> result,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric) {
  *result = upper_spatial_metric;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      result->get(i, j) -=
          unit_normal_vector.get(i) * unit_normal_vector.get(j);
    }
  }
}

template <typename Frame>
tnsr::II<DataVector, 3, Frame> inverse_surface_metric(
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric) {
  tnsr::II<DataVector, 3, Frame> result{};
  inverse_surface_metric(make_not_null(&result), unit_normal_vector,
                         upper_spatial_metric);
  return result;
}
}  // namespace StrahlkorperGr

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                             \
  template void StrahlkorperGr::inverse_surface_metric<FRAME(data)>(     \
      const gsl::not_null<tnsr::II<DataVector, 3, FRAME(data)>*> result, \
      const tnsr::I<DataVector, 3, FRAME(data)>& unit_normal_vector,     \
      const tnsr::II<DataVector, 3, FRAME(data)>& upper_spatial_metric); \
  template tnsr::II<DataVector, 3, FRAME(data)>                          \
  StrahlkorperGr::inverse_surface_metric<FRAME(data)>(                   \
      const tnsr::I<DataVector, 3, FRAME(data)>& unit_normal_vector,     \
      const tnsr::II<DataVector, 3, FRAME(data)>& upper_spatial_metric);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
