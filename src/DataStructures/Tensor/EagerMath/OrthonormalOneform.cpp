// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/EagerMath/OrthonormalOneform.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <typename DataType, size_t VolumeDim, typename Frame>
void orthonormal_oneform(
    const gsl::not_null<tnsr::i<DataType, VolumeDim, Frame>*> orthonormal_form,
    const tnsr::i<DataType, VolumeDim, Frame>& unit_form,
    const tnsr::II<DataType, VolumeDim, Frame>& inv_spatial_metric) noexcept {
  *orthonormal_form = unit_form;
  const size_t number_of_points = get_size(get<0>(unit_form));
  for (size_t s = 0; s < number_of_points; ++s) {
    size_t min_index = 0;
    for (size_t i = 1; i < VolumeDim; ++i) {
      if (std::abs(get_element(unit_form.get(i), s)) <
          std::abs(get_element(unit_form.get(min_index), s))) {
        min_index = i;
      }
    }

    double proj = 0.0;
    for (size_t i = 0; i < VolumeDim; ++i) {
      proj += (get_element(inv_spatial_metric.get(min_index, i), s) *
               get_element(unit_form.get(i), s));
    }
    const double inv_magnitude =
        1.0 /
        sqrt(get_element(inv_spatial_metric.get(min_index, min_index), s) -
             square(proj));

    proj *= -inv_magnitude;
    for (size_t i = 0; i < VolumeDim; ++i) {
      get_element(orthonormal_form->get(i), s) *= proj;
      if (i == min_index) {
        get_element(orthonormal_form->get(i), s) += inv_magnitude;
      }
    }
  }
}

template <typename DataType, size_t VolumeDim, typename Frame>
tnsr::i<DataType, VolumeDim, Frame> orthonormal_oneform(
    const tnsr::i<DataType, VolumeDim, Frame>& unit_form,
    const tnsr::II<DataType, VolumeDim, Frame>& inv_spatial_metric) noexcept {
  tnsr::i<DataType, VolumeDim, Frame> orthonormal_form{};
  orthonormal_oneform(make_not_null(&orthonormal_form), unit_form,
                      inv_spatial_metric);
  return orthonormal_form;
}

template <typename DataType, typename Frame>
void orthonormal_oneform(
    gsl::not_null<tnsr::i<DataType, 3, Frame>*> orthonormal_form,
    const tnsr::i<DataType, 3, Frame>& first_unit_form,
    const tnsr::i<DataType, 3, Frame>& second_unit_form,
    const tnsr::ii<DataType, 3, Frame>& spatial_metric,
    const Scalar<DataType>& det_spatial_metric) noexcept {
  *orthonormal_form = cross_product(first_unit_form, second_unit_form,
                                    spatial_metric, det_spatial_metric);
}

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame> orthonormal_oneform(
    const tnsr::i<DataType, 3, Frame>& first_unit_form,
    const tnsr::i<DataType, 3, Frame>& second_unit_form,
    const tnsr::ii<DataType, 3, Frame>& spatial_metric,
    const Scalar<DataType>& det_spatial_metric) noexcept {
  tnsr::i<DataType, 3, Frame> orthonormal_form{};
  orthonormal_oneform(make_not_null(&orthonormal_form), first_unit_form,
                      second_unit_form, spatial_metric, det_spatial_metric);
  return orthonormal_form;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE_FIRST_FORM(_, data)                                      \
  template void orthonormal_oneform(                                         \
      const gsl::not_null<tnsr::i<DTYPE(data), DIM(data), FRAME(data)>*>     \
          orthonormal_form,                                                  \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& unit_form,         \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inv_spatial_metric) noexcept;                                      \
  template tnsr::i<DTYPE(data), DIM(data), FRAME(data)> orthonormal_oneform( \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& unit_form,         \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inv_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_FIRST_FORM, (double, DataVector),
                        (Frame::Inertial), (2, 3))

#define INSTANTIATE_SECOND_FORM(_, data)                             \
  template void orthonormal_oneform(                                 \
      const gsl::not_null<tnsr::i<DTYPE(data), 3, FRAME(data)>*>     \
          orthonormal_form,                                          \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& first_unit_form,   \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& second_unit_form,  \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& spatial_metric,   \
      const Scalar<DTYPE(data)>& det_spatial_metric) noexcept;       \
  template tnsr::i<DTYPE(data), 3, FRAME(data)> orthonormal_oneform( \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& first_unit_form,   \
      const tnsr::i<DTYPE(data), 3, FRAME(data)>& second_unit_form,  \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& spatial_metric,   \
      const Scalar<DTYPE(data)>& det_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SECOND_FORM, (double, DataVector),
                        (Frame::Inertial))

#undef INSTANTIATE_SECOND_FORM
#undef INSTANTIATE_FIRST_FORM
#undef DIM
#undef FRAME
#undef DTYPE
/// \endcond
