// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace GeneralizedHarmonic {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> three_index_constraint(
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  auto constraint =
      make_with_value<tnsr::iaa<DataType, SpatialDim, Frame>>(phi, 0.0);
  three_index_constraint<SpatialDim, Frame, DataType>(&constraint,
                                                      d_spacetime_metric, phi);
  return constraint;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void three_index_constraint(
    gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  // Declare iterators for d_spacetime_metric and phi outside the for loop,
  // because they are const but constraint is not
  auto d_spacetime_metric_it = d_spacetime_metric.begin(), phi_it = phi.begin();

  for (auto constraint_it = (*constraint).begin();
       constraint_it != (*constraint).end();
       ++constraint_it, (void)++d_spacetime_metric_it, (void)++phi_it) {
    *constraint_it = *d_spacetime_metric_it - *phi_it;
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> gauge_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  auto constraint =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame>>(pi, 0.0);
  gauge_constraint<SpatialDim, Frame, DataType>(
      &constraint, gauge_function, spacetime_normal_one_form,
      spacetime_normal_vector, inverse_spatial_metric,
      inverse_spacetime_metric, pi, phi);
  return constraint;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void gauge_constraint(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    (*constraint).get(a) = gauge_function.get(a);
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        (*constraint).get(a) +=
            inverse_spatial_metric.get(i, j) * phi.get(i, j + 1, a);
      }
    }
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      (*constraint).get(a) += spacetime_normal_vector.get(b) * pi.get(b, a);
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        (*constraint).get(a) -= 0.5 * spacetime_normal_one_form.get(a) *
                                   pi.get(b, c) *
                                   inverse_spacetime_metric.get(b, c);
        if (a > 0) {
          (*constraint).get(a) -=
              0.5 * phi.get(a - 1, b, c) * inverse_spacetime_metric.get(b, c);
        }
        for (size_t i = 0; i < SpatialDim; ++i) {
          (*constraint).get(a) -= 0.5 * spacetime_normal_one_form.get(a) *
                                     spacetime_normal_vector.get(i + 1) *
                                     phi.get(i, b, c) *
                                     inverse_spacetime_metric.get(b, c);
        }
      }
    }
  }
}
}  // namespace GeneralizedHarmonic

// Explicit Instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>                  \
  GeneralizedHarmonic::three_index_constraint(                             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_spacetime_metric,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept; \
  template void GeneralizedHarmonic::three_index_constraint(               \
      gsl::not_null<tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>*>       \
          constraint,                                                      \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                \
          d_spacetime_metric,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept; \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                    \
  GeneralizedHarmonic::gauge_constraint(                                   \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,  \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spacetime_normal_one_form,                                       \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spacetime_normal_vector,                                         \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric,                                          \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spacetime_metric,                                        \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept; \
  template void GeneralizedHarmonic::gauge_constraint(                     \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>   \
          constraint,                                                      \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,  \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spacetime_normal_one_form,                                       \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spacetime_normal_vector,                                         \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric,                                          \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spacetime_metric,                                        \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
