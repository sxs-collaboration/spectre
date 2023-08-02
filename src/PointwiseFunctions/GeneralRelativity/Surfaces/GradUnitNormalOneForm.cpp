// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/GradUnitNormalOneForm.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace StrahlkorperGr {
template <typename Frame>
void grad_unit_normal_one_form(
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame>*> result,
    const tnsr::i<DataVector, 3, Frame>& r_hat,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, 3, Frame>& d2x_radius,
    const DataVector& one_over_one_form_magnitude,
    const tnsr::Ijj<DataVector, 3, Frame>& christoffel_2nd_kind) {
  const DataVector one_over_radius = 1.0 / get(radius);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // symmetry
      result->get(i, j) = -one_over_one_form_magnitude *
                          (r_hat.get(i) * r_hat.get(j) * one_over_radius +
                           d2x_radius.get(i, j));
      for (size_t k = 0; k < 3; ++k) {
        result->get(i, j) -=
            unit_normal_one_form.get(k) * christoffel_2nd_kind.get(k, i, j);
      }
    }
    result->get(i, i) += one_over_radius * one_over_one_form_magnitude;
  }
}

template <typename Frame>
tnsr::ii<DataVector, 3, Frame> grad_unit_normal_one_form(
    const tnsr::i<DataVector, 3, Frame>& r_hat,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, 3, Frame>& d2x_radius,
    const DataVector& one_over_one_form_magnitude,
    const tnsr::Ijj<DataVector, 3, Frame>& christoffel_2nd_kind) {
  tnsr::ii<DataVector, 3, Frame> result{};
  grad_unit_normal_one_form(make_not_null(&result), r_hat, radius,
                            unit_normal_one_form, d2x_radius,
                            one_over_one_form_magnitude, christoffel_2nd_kind);
  return result;
}
}  // namespace StrahlkorperGr

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                              \
  template void StrahlkorperGr::grad_unit_normal_one_form<FRAME(data)>(   \
      gsl::not_null<tnsr::ii<DataVector, 3, FRAME(data)>*> result,        \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat,                   \
      const Scalar<DataVector>& radius,                                   \
      const tnsr::i<DataVector, 3, FRAME(data)>& unit_normal_one_form,    \
      const tnsr::ii<DataVector, 3, FRAME(data)>& d2x_radius,             \
      const DataVector& one_over_one_form_magnitude,                      \
      const tnsr::Ijj<DataVector, 3, FRAME(data)>& christoffel_2nd_kind); \
  template tnsr::ii<DataVector, 3, FRAME(data)>                           \
  StrahlkorperGr::grad_unit_normal_one_form<FRAME(data)>(                 \
      const tnsr::i<DataVector, 3, FRAME(data)>& r_hat,                   \
      const Scalar<DataVector>& radius,                                   \
      const tnsr::i<DataVector, 3, FRAME(data)>& unit_normal_one_form,    \
      const tnsr::ii<DataVector, 3, FRAME(data)>& d2x_radius,             \
      const DataVector& one_over_one_form_magnitude,                      \
      const tnsr::Ijj<DataVector, 3, FRAME(data)>& christoffel_2nd_kind);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
