// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/UnitNormalOneForm.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace StrahlkorperGr {
template <typename Frame>
void unit_normal_one_form(
    const gsl::not_null<tnsr::i<DataVector, 3, Frame>*> result,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const DataVector& one_over_one_form_magnitude) {
  *result = normal_one_form;
  for (size_t i = 0; i < 3; ++i) {
    result->get(i) *= one_over_one_form_magnitude;
  }
}

template <typename Frame>
tnsr::i<DataVector, 3, Frame> unit_normal_one_form(
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const DataVector& one_over_one_form_magnitude) {
  tnsr::i<DataVector, 3, Frame> result{};
  unit_normal_one_form(make_not_null(&result), normal_one_form,
                       one_over_one_form_magnitude);
  return result;
}
}  // namespace StrahlkorperGr

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                            \
  template void StrahlkorperGr::unit_normal_one_form<FRAME(data)>(      \
      const gsl::not_null<tnsr::i<DataVector, 3, FRAME(data)>*> result, \
      const tnsr::i<DataVector, 3, FRAME(data)>& normal_one_form,       \
      const DataVector& one_over_one_form_magnitude);                   \
  template tnsr::i<DataVector, 3, FRAME(data)>                          \
  StrahlkorperGr::unit_normal_one_form<FRAME(data)>(                    \
      const tnsr::i<DataVector, 3, FRAME(data)>& normal_one_form,       \
      const DataVector& one_over_one_form_magnitude);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
