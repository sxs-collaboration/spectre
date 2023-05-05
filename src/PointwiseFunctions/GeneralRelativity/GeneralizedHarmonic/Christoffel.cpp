// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gh {
template <typename DataType, size_t SpatialDim, typename Frame>
void christoffel_second_kind(
    const gsl::not_null<tnsr::Ijj<DataType, SpatialDim, Frame>*> christoffel,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inv_metric) {
  destructive_resize_components(christoffel, get_size(*phi.begin()));
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      for (size_t m = 0; m < SpatialDim; ++m) {
        christoffel->get(m, i, j) =
            0.5 * inv_metric.get(m, 0) *
            (phi.get(i, j + 1, 1) + phi.get(j, i + 1, 1) -
             phi.get(0, i + 1, j + 1));
        for (size_t k = 1; k < SpatialDim; ++k) {
          christoffel->get(m, i, j) +=
              0.5 * inv_metric.get(m, k) *
              (phi.get(i, j + 1, k + 1) + phi.get(j, i + 1, k + 1) -
               phi.get(k, i + 1, j + 1));
        }
      }
    }
  }
}
template <typename DataType, size_t SpatialDim, typename Frame>
auto christoffel_second_kind(
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inv_metric)
    -> tnsr::Ijj<DataType, SpatialDim, Frame> {
  tnsr::Ijj<DataType, SpatialDim, Frame> christoffel(
      get_size(get<0, 0, 0>(phi)));
  christoffel_second_kind(make_not_null(&christoffel), phi, inv_metric);
  return christoffel;
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::a<DataType, SpatialDim, Frame> trace_christoffel(
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) {
  auto trace = make_with_value<tnsr::a<DataType, SpatialDim, Frame>>(pi, 0.0);
  trace_christoffel<DataType, SpatialDim, Frame>(
      &trace, spacetime_normal_one_form, spacetime_normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi);
  return trace;
}

template <typename DataType, size_t SpatialDim, typename Frame>
void trace_christoffel(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> trace,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) {
  destructive_resize_components(trace,
                                get_size(get<0>(spacetime_normal_one_form)));
  get<0>(*trace) = 0.0;
  // Compute common terms between components.
  for (size_t b = 0; b < SpatialDim + 1; ++b) {
    get<0>(*trace) -= 0.5 * pi.get(b, b) * inverse_spacetime_metric.get(b, b);
    for (size_t i = 0; i < SpatialDim; ++i) {
      get<0>(*trace) -= 0.5 * spacetime_normal_vector.get(i + 1) *
                        phi.get(i, b, b) * inverse_spacetime_metric.get(b, b);
    }
    for (size_t c = b + 1; c < SpatialDim + 1; ++c) {
      get<0>(*trace) -= pi.get(b, c) * inverse_spacetime_metric.get(b, c);
      for (size_t i = 0; i < SpatialDim; ++i) {
        get<0>(*trace) -= spacetime_normal_vector.get(i + 1) *
                          phi.get(i, b, c) * inverse_spacetime_metric.get(b, c);
      }
    }
  }

  // Compute spatial components
  for (size_t a = 1; a < SpatialDim + 1; ++a) {
    trace->get(a) = spacetime_normal_one_form.get(a) * get<0>(*trace);
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        trace->get(a) +=
            inverse_spatial_metric.get(i, j) * phi.get(i, j + 1, a);
      }
    }
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      trace->get(a) += spacetime_normal_vector.get(b) * pi.get(b, a);
      // a is always > 0 in this case so delta^i_a is taken care of.
      trace->get(a) -=
          0.5 * phi.get(a - 1, b, b) * inverse_spacetime_metric.get(b, b);
      for (size_t c = b + 1; c < SpatialDim + 1; ++c) {
        trace->get(a) -=
            phi.get(a - 1, b, c) * inverse_spacetime_metric.get(b, c);
      }
    }
  }
  // Now set 0 component
  get<0>(*trace) *= get<0>(spacetime_normal_one_form);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      get<0>(*trace) += inverse_spatial_metric.get(i, j) * phi.get(i, j + 1, 0);
    }
  }
  for (size_t b = 0; b < SpatialDim + 1; ++b) {
    get<0>(*trace) += spacetime_normal_vector.get(b) * pi.get(b, 0);
  }
}
}  // namespace gh

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                   \
  template void gh::christoffel_second_kind(                                   \
      const gsl::not_null<tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>*>     \
          christoffel,                                                         \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,               \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>& inv_metric);        \
  template tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>                      \
  gh::christoffel_second_kind(                                                 \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,               \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>& inv_metric);        \
  template void gh::trace_christoffel(                                         \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>       \
          trace,                                                               \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                      \
          spacetime_normal_one_form,                                           \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                      \
          spacetime_normal_vector,                                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spatial_metric,                                              \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spacetime_metric,                                            \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                 \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);              \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)> gh::trace_christoffel( \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                      \
          spacetime_normal_one_form,                                           \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                      \
          spacetime_normal_vector,                                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spatial_metric,                                              \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spacetime_metric,                                            \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                 \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial,
                         Frame::Spherical<Frame::Inertial>,
                         Frame::Spherical<Frame::Grid>))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
