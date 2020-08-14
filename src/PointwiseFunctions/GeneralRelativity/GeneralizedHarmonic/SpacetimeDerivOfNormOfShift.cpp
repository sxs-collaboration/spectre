// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivOfNormOfShift.hpp"

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLowerShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace GeneralizedHarmonic {
namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
struct D4NormOfShiftBuffer;

template <size_t SpatialDim, typename Frame>
struct D4NormOfShiftBuffer<SpatialDim, Frame, double> {
  explicit D4NormOfShiftBuffer(const size_t /*size*/) noexcept {}

  tnsr::i<double, SpatialDim, Frame> lower_shift{};
  tnsr::iJ<double, SpatialDim, Frame> deriv_shift{};
  tnsr::i<double, SpatialDim, Frame> dt_lower_shift{};
  tnsr::I<double, SpatialDim, Frame> dt_shift{};
};

template <size_t SpatialDim, typename Frame>
struct D4NormOfShiftBuffer<SpatialDim, Frame, DataVector> {
 private:
  // We make one giant allocation so that we don't thrash the heap.
  Variables<tmpl::list<::Tags::Tempi<0, SpatialDim, Frame, DataVector>,
                       ::Tags::TempiJ<1, SpatialDim, Frame, DataVector>,
                       ::Tags::Tempi<2, SpatialDim, Frame, DataVector>,
                       ::Tags::TempI<3, SpatialDim, Frame, DataVector>>>
      buffer_;

 public:
  explicit D4NormOfShiftBuffer(const size_t size) noexcept
      : buffer_(size),
        lower_shift(
            get<::Tags::Tempi<0, SpatialDim, Frame, DataVector>>(buffer_)),
        deriv_shift(
            get<::Tags::TempiJ<1, SpatialDim, Frame, DataVector>>(buffer_)),
        dt_lower_shift(
            get<::Tags::Tempi<2, SpatialDim, Frame, DataVector>>(buffer_)),
        dt_shift(
            get<::Tags::TempI<3, SpatialDim, Frame, DataVector>>(buffer_)) {}

  tnsr::i<DataVector, SpatialDim, Frame>& lower_shift;
  tnsr::iJ<DataVector, SpatialDim, Frame>& deriv_shift;
  tnsr::i<DataVector, SpatialDim, Frame>& dt_lower_shift;
  tnsr::I<DataVector, SpatialDim, Frame>& dt_shift;
};
}  // namespace

template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_norm_of_shift(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_norm_of_shift,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  if (UNLIKELY(get_size(get<0>(*d4_norm_of_shift)) != get_size(get(lapse)))) {
    *d4_norm_of_shift =
        tnsr::a<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  // Use a Variables to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  D4NormOfShiftBuffer<SpatialDim, Frame, DataType> buffer(get_size(get(lapse)));

  raise_or_lower_index(make_not_null(&buffer.lower_shift), shift,
                       spatial_metric);
  spatial_deriv_of_shift(make_not_null(&buffer.deriv_shift), lapse,
                         inverse_spacetime_metric, spacetime_unit_normal, phi);
  time_deriv_of_lower_shift(make_not_null(&buffer.dt_lower_shift), lapse, shift,
                            spatial_metric, spacetime_unit_normal, phi, pi);
  time_deriv_of_shift(make_not_null(&buffer.dt_shift), lapse, shift,
                      inverse_spatial_metric, spacetime_unit_normal, phi, pi);
  // first term for component 0
  get<0>(*d4_norm_of_shift) =
      shift.get(0) * buffer.dt_lower_shift.get(0) +
      buffer.lower_shift.get(0) * buffer.dt_shift.get(0);
  for (size_t i = 1; i < SpatialDim; ++i) {
    get<0>(*d4_norm_of_shift) +=
        shift.get(i) * buffer.dt_lower_shift.get(i) +
        buffer.lower_shift.get(i) * buffer.dt_shift.get(i);
  }
  // second term for components 1,2,3
  for (size_t j = 0; j < SpatialDim; ++j) {
    d4_norm_of_shift->get(1 + j) =
        shift.get(0) * phi.get(j, 0, 1) +
        buffer.lower_shift.get(0) * buffer.deriv_shift.get(j, 0);
    for (size_t i = 1; i < SpatialDim; ++i) {
      d4_norm_of_shift->get(1 + j) +=
          shift.get(i) * phi.get(j, 0, i + 1) +
          buffer.lower_shift.get(i) * buffer.deriv_shift.get(j, i);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_norm_of_shift(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  tnsr::a<DataType, SpatialDim, Frame> d4_norm_of_shift{};
  GeneralizedHarmonic::spacetime_deriv_of_norm_of_shift<SpatialDim, Frame,
                                                        DataType>(
      make_not_null(&d4_norm_of_shift), lapse, shift, spatial_metric,
      inverse_spatial_metric, inverse_spacetime_metric, spacetime_unit_normal,
      phi, pi);
  return d4_norm_of_shift;
}
}  // namespace GeneralizedHarmonic

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void GeneralizedHarmonic::spacetime_deriv_of_norm_of_shift(     \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>   \
          d4_norm_of_shift,                                                \
      const Scalar<DTYPE(data)>& lapse,                                    \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric, \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric,                                          \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spacetime_metric,                                        \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spacetime_unit_normal,                                           \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;   \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                    \
  GeneralizedHarmonic::spacetime_deriv_of_norm_of_shift(                   \
      const Scalar<DTYPE(data)>& lapse,                                    \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric, \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric,                                          \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spacetime_metric,                                        \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spacetime_unit_normal,                                           \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
