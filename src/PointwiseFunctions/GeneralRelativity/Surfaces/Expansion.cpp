// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/Expansion.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace StrahlkorperGr {
template <typename Frame>
void expansion(const gsl::not_null<Scalar<DataVector>*> result,
               const tnsr::ii<DataVector, 3, Frame>& grad_normal,
               const tnsr::II<DataVector, 3, Frame>& inverse_surface_metric,
               const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature) {
  // If you want the future *ingoing* null expansion,
  // the formula is the same as here except you
  // change the sign on grad_normal just before you
  // subtract the extrinsic curvature.
  // That is, if GsBar is the value of grad_normal
  // at this point in the code, and S^i is the unit
  // spatial normal to the surface,
  // the outgoing expansion is
  // (g^ij - S^i S^j) (GsBar_ij - K_ij)
  // and the ingoing expansion is
  // (g^ij - S^i S^j) (-GsBar_ij - K_ij)
  set_number_of_grid_points(result, grad_normal);
  get(*result) = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(*result) += inverse_surface_metric.get(i, j) *
                      (grad_normal.get(i, j) - extrinsic_curvature.get(i, j));
    }
  }
}

template <typename Frame>
Scalar<DataVector> expansion(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::II<DataVector, 3, Frame>& inverse_surface_metric,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature) {
  Scalar<DataVector> result{};
  expansion(make_not_null(&result), grad_normal, inverse_surface_metric,
            extrinsic_curvature);
  return result;
}
}  // namespace StrahlkorperGr

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                              \
  template void StrahlkorperGr::expansion<FRAME(data)>(                   \
      const gsl::not_null<Scalar<DataVector>*> result,                    \
      const tnsr::ii<DataVector, 3, FRAME(data)>& grad_normal,            \
      const tnsr::II<DataVector, 3, FRAME(data)>& inverse_surface_metric, \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature);   \
  template Scalar<DataVector> StrahlkorperGr::expansion<FRAME(data)>(     \
      const tnsr::ii<DataVector, 3, FRAME(data)>& grad_normal,            \
      const tnsr::II<DataVector, 3, FRAME(data)>& inverse_surface_metric, \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
