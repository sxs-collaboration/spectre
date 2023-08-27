// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/ExtrinsicCurvature.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr::surfaces {
template <typename Frame>
void extrinsic_curvature(
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame>*> result,
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector) {
  Scalar<DataVector> nI_nJ_gradnij(get<0, 0>(grad_normal).size(), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(nI_nJ_gradnij) += unit_normal_vector.get(i) *
                            unit_normal_vector.get(j) * grad_normal.get(i, j);
    }
  }

  *result = grad_normal;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result->get(i, j) += unit_normal_one_form.get(i) *
                           unit_normal_one_form.get(j) * get(nI_nJ_gradnij);
      for (size_t k = 0; k < 3; ++k) {
        result->get(i, j) -=
            unit_normal_vector.get(k) *
            (unit_normal_one_form.get(i) * grad_normal.get(j, k) +
             unit_normal_one_form.get(j) * grad_normal.get(i, k));
      }
    }
  }
}

template <typename Frame>
tnsr::ii<DataVector, 3, Frame> extrinsic_curvature(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector) {
  tnsr::ii<DataVector, 3, Frame> result{};
  extrinsic_curvature(make_not_null(&result), grad_normal, unit_normal_one_form,
                      unit_normal_vector);
  return result;
}
}  // namespace gr::surfaces

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                             \
  template void gr::surfaces::extrinsic_curvature<FRAME(data)>(          \
      const gsl::not_null<tnsr::ii<DataVector, 3, FRAME(data)>*> result, \
      const tnsr::ii<DataVector, 3, FRAME(data)>& grad_normal,           \
      const tnsr::i<DataVector, 3, FRAME(data)>& unit_normal_one_form,   \
      const tnsr::I<DataVector, 3, FRAME(data)>& unit_normal_vector);    \
  template tnsr::ii<DataVector, 3, FRAME(data)>                          \
  gr::surfaces::extrinsic_curvature<FRAME(data)>(                        \
      const tnsr::ii<DataVector, 3, FRAME(data)>& grad_normal,           \
      const tnsr::i<DataVector, 3, FRAME(data)>& unit_normal_one_form,   \
      const tnsr::I<DataVector, 3, FRAME(data)>& unit_normal_vector);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
