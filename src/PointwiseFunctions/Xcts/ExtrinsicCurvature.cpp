// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/ExtrinsicCurvature.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {

template <typename DataType>
void extrinsic_curvature(
    const gsl::not_null<tnsr::ii<DataType, 3>*> result,
    const Scalar<DataType>& conformal_factor, const Scalar<DataType>& lapse,
    const tnsr::ii<DataType, 3>& conformal_metric,
    const tnsr::II<DataType, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataType>& trace_extrinsic_curvature) {
  tenex::evaluate<ti::i, ti::j>(
      result,
      pow<4>(conformal_factor()) *
          (conformal_metric(ti::i, ti::k) * conformal_metric(ti::j, ti::l) *
               longitudinal_shift_minus_dt_conformal_metric(ti::K, ti::L) /
               (2. * lapse()) +
           conformal_metric(ti::i, ti::j) * trace_extrinsic_curvature() / 3.));
}

template <typename DataType>
tnsr::ii<DataType, 3> extrinsic_curvature(
    const Scalar<DataType>& conformal_factor, const Scalar<DataType>& lapse,
    const tnsr::ii<DataType, 3>& conformal_metric,
    const tnsr::II<DataType, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataType>& trace_extrinsic_curvature) {
  tnsr::ii<DataType, 3> result{get_size(get(conformal_factor))};
  extrinsic_curvature(
      make_not_null(&result), conformal_factor, lapse, conformal_metric,
      longitudinal_shift_minus_dt_conformal_metric, trace_extrinsic_curvature);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                 \
  template void extrinsic_curvature(                         \
      const gsl::not_null<tnsr::ii<DTYPE(data), 3>*> result, \
      const Scalar<DTYPE(data)>& conformal_factor,           \
      const Scalar<DTYPE(data)>& lapse,                      \
      const tnsr::ii<DTYPE(data), 3>& conformal_metric,      \
      const tnsr::II<DTYPE(data), 3>&                        \
          longitudinal_shift_minus_dt_conformal_metric,      \
      const Scalar<DTYPE(data)>& trace_extrinsic_curvature); \
  template tnsr::ii<DTYPE(data), 3> extrinsic_curvature(     \
      const Scalar<DTYPE(data)>& conformal_factor,           \
      const Scalar<DTYPE(data)>& lapse,                      \
      const tnsr::ii<DTYPE(data), 3>& conformal_metric,      \
      const tnsr::II<DTYPE(data), 3>&                        \
          longitudinal_shift_minus_dt_conformal_metric,      \
      const Scalar<DTYPE(data)>& trace_extrinsic_curvature);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace Xcts
