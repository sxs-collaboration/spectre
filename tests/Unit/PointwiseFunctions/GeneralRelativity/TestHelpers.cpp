// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"

#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

template <typename DataType>
Scalar<DataType> make_lapse(const DataType& used_for_size) {
  return Scalar<DataType>{make_with_value<DataType>(used_for_size, 3.)};
}

template <size_t SpatialDim, typename DataType>
tnsr::I<DataType, SpatialDim> make_shift(const DataType& used_for_size) {
  auto shift =
      make_with_value<tnsr::I<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    shift.get(i) = make_with_value<DataType>(used_for_size, i + 1.);
  }
  return shift;
}

template <size_t SpatialDim, typename DataType>
tnsr::i<DataType, SpatialDim> make_lower_shift(const DataType& used_for_size) {
  auto shift =
      make_with_value<tnsr::i<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    shift.get(i) = make_with_value<DataType>(used_for_size, i + 1.);
  }
  return shift;
}

template <size_t SpatialDim, typename DataType>
tnsr::ii<DataType, SpatialDim> make_spatial_metric(
    const DataType& used_for_size) {
  auto metric =
      make_with_value<tnsr::ii<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      metric.get(i, j) =
          make_with_value<DataType>(used_for_size, (i + 1.) * (j + 1.));
    }
  }
  return metric;
}

template <size_t SpatialDim, typename DataType>
tnsr::II<DataType, SpatialDim> make_inverse_spatial_metric(
    const DataType& used_for_size) {
  auto metric =
      make_with_value<tnsr::II<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      metric.get(i, j) =
          make_with_value<DataType>(used_for_size, (i + 1.) * (j + 1.));
    }
  }
  return metric;
}

template <size_t SpatialDim, typename DataType>
tnsr::A<DataType, SpatialDim> make_dummy_vector(const DataType& used_for_size) {
  auto result =
      make_with_value<tnsr::A<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    result.get(a) = make_with_value<DataType>(used_for_size, a + 1.);
  }
  return result;
}

template <size_t SpatialDim, typename DataType>
tnsr::a<DataType, SpatialDim> make_dummy_one_form(
    const DataType& used_for_size) {
  auto one_form =
      make_with_value<tnsr::a<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    one_form.get(a) = make_with_value<DataType>(used_for_size, a + 1.);
  }
  return one_form;
}

template <typename DataType>
Scalar<DataType> make_dt_lapse(const DataType& used_for_size) {
  return Scalar<DataType>{make_with_value<DataType>(used_for_size, 5.)};
}

template <size_t SpatialDim, typename DataType>
tnsr::I<DataType, SpatialDim> make_dt_shift(const DataType& used_for_size) {
  auto dt_shift =
      make_with_value<tnsr::I<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    dt_shift.get(i) =
        make_with_value<DataType>(used_for_size, SpatialDim * (i + 1.));
  }
  return dt_shift;
}

template <size_t SpatialDim, typename DataType>
tnsr::ii<DataType, SpatialDim> make_dt_spatial_metric(
    const DataType& used_for_size) {
  auto dt_metric =
      make_with_value<tnsr::ii<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      dt_metric.get(i, j) = make_with_value<DataType>(used_for_size, i + j);
    }
  }
  return dt_metric;
}

template <size_t SpatialDim, typename DataType>
tnsr::i<DataType, SpatialDim> make_deriv_lapse(const DataType& used_for_size) {
  auto deriv_lapse =
      make_with_value<tnsr::i<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    deriv_lapse.get(i) =
        make_with_value<DataType>(used_for_size, 2.5 * (i + 1.));
  }
  return deriv_lapse;
}

template <size_t SpatialDim, typename DataType>
tnsr::iJ<DataType, SpatialDim> make_deriv_shift(const DataType& used_for_size) {
  auto deriv_shift =
      make_with_value<tnsr::iJ<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      deriv_shift.get(i, j) =
          make_with_value<DataType>(used_for_size, 3. * (j + 1.) - i + 4.);
    }
  }
  return deriv_shift;
}

template <size_t SpatialDim, typename DataType>
tnsr::ijj<DataType, SpatialDim> make_deriv_spatial_metric(
    const DataType& used_for_size) {
  auto deriv_spatial_metric =
      make_with_value<tnsr::ijj<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        deriv_spatial_metric.get(k, i, j) = make_with_value<DataType>(
            used_for_size, 3. * (i + 1.) * (j + 1.) + k);
      }
    }
  }
  return deriv_spatial_metric;
}

template <size_t SpatialDim, typename DataType>
tnsr::abb<DataType, SpatialDim> make_spacetime_deriv_spacetime_metric(
    const DataType& used_for_size) {
  auto d_spacetime_metric =
      make_with_value<tnsr::abb<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim + 1; i++) {
    for (size_t j = i; j < SpatialDim + 1; j++) {
      for (size_t k = 0; k < SpatialDim + 1; k++) {
        d_spacetime_metric.get(k, i, j) = make_with_value<DataType>(
            used_for_size, (i + 1.) * (j + 1.) * (k + 3.));
      }
    }
  }
  return d_spacetime_metric;
}

template <size_t SpatialDim, typename DataType>
tnsr::iaa<DataType, SpatialDim> make_spatial_deriv_spacetime_metric(
    const DataType& used_for_size) {
  auto d_spacetime_metric =
      make_with_value<tnsr::iaa<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim + 1; i++) {
    for (size_t j = i; j < SpatialDim + 1; j++) {
      for (size_t k = 0; k < SpatialDim; k++) {
        d_spacetime_metric.get(k, i, j) = make_with_value<DataType>(
            used_for_size, (i + 1.) * (j + 1.) * (k + 3.));
      }
    }
  }
  return d_spacetime_metric;
}

template <size_t SpatialDim, typename DataType>
tnsr::aa<DataType, SpatialDim> make_dt_spacetime_metric(
    const DataType& used_for_size) {
  auto dt_spacetime_metric =
      make_with_value<tnsr::aa<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim + 1; i++) {
    for (size_t j = i; j < SpatialDim + 1; j++) {
      dt_spacetime_metric.get(i, j) =
          make_with_value<DataType>(used_for_size, i * j + 0.5);
    }
  }
  return dt_spacetime_metric;
}

template <size_t SpatialDim, typename DataType>
tnsr::Abb<DataType, SpatialDim> make_spacetime_christoffel_second_kind(
    const DataType& used_for_size) {
  auto christoffel =
      make_with_value<tnsr::Abb<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim + 1; i++) {
    for (size_t j = i; j < SpatialDim + 1; j++) {
      for (size_t k = 0; k < SpatialDim + 1; k++) {
        christoffel.get(k, i, j) = make_with_value<DataType>(
            used_for_size, (i + 2.) * (j + 1.) * (k + 2.));
      }
    }
  }
  return christoffel;
}

template <size_t SpatialDim, typename DataType>
tnsr::Ijj<DataType, SpatialDim> make_spatial_christoffel_second_kind(
    const DataType& used_for_size) {
  auto christoffel =
      make_with_value<tnsr::Ijj<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = i; j < SpatialDim; j++) {
      for (size_t k = 0; k < SpatialDim; k++) {
        christoffel.get(k, i, j) = make_with_value<DataType>(
            used_for_size, 4. * (i + 1.) * (j + 1.) - k);
      }
    }
  }
  return christoffel;
}

template <size_t SpatialDim, typename DataType>
tnsr::aBcc<DataType, SpatialDim> make_deriv_spacetime_christoffel_second_kind(
    const DataType& used_for_size) {
  auto deriv_christoffel =
      make_with_value<tnsr::aBcc<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim + 1; i++) {
    for (size_t j = i; j < SpatialDim + 1; j++) {
      for (size_t k = 0; k < SpatialDim + 1; k++) {
        for (size_t l = 0; l < SpatialDim + 1; ++l) {
          deriv_christoffel.get(k, l, i, j) = make_with_value<DataType>(
              used_for_size, (i + 2.) * (j + 1.) * (k + 2.) * (l + 4.));
        }
      }
    }
  }
  return deriv_christoffel;
}

template <size_t SpatialDim, typename DataType>
tnsr::iJkk<DataType, SpatialDim> make_deriv_spatial_christoffel_second_kind(
    const DataType& used_for_size){
  auto deriv_christoffel =
      make_with_value<tnsr::iJkk<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = i; j < SpatialDim; j++) {
      for (size_t k = 0; k < SpatialDim; k++) {
        for (size_t l = 0; l < SpatialDim; ++l) {
          deriv_christoffel.get(k, l, i, j) = make_with_value<DataType>(
              used_for_size, 4. * (i + 1.) * (j + 1.) - k * (l + 4.));
        }
      }
    }
  }
  return deriv_christoffel;
}


template <size_t SpatialDim, typename DataType>
tnsr::aa<DataType, SpatialDim> make_spacetime_metric(
    const DataType& used_for_size) {
  auto spacetime_metric =
      make_with_value<tnsr::aa<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t mu = 0; mu < SpatialDim + 1; ++mu) {
    for (size_t nu = mu; nu < SpatialDim + 1; ++nu) {
      spacetime_metric.get(mu, nu) =
          make_with_value<DataType>(used_for_size, -2. * (mu + 2.) * (nu + 2.));
    }
  }
  return spacetime_metric;
}

template <size_t SpatialDim, typename DataType>
tnsr::AA<DataType, SpatialDim> make_inverse_spacetime_metric(
    const DataType& used_for_size) {
  auto inverse_spacetime_metric =
      make_with_value<tnsr::AA<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t mu = 0; mu < SpatialDim + 1; ++mu) {
    for (size_t nu = mu; nu < SpatialDim + 1; ++nu) {
      inverse_spacetime_metric.get(mu, nu) =
          make_with_value<DataType>(used_for_size, -2. * (mu + 2.) * (nu + 2.));
    }
  }
  return inverse_spacetime_metric;
}

template <typename DataType>
Scalar<DataType> make_trace_extrinsic_curvature(const DataType& used_for_size) {
  return Scalar<DataType>{make_with_value<DataType>(used_for_size, 5.)};
}

template <size_t SpatialDim, typename DataType>
tnsr::i<DataType, SpatialDim> make_trace_spatial_christoffel_first_kind(
    const DataType& used_for_size) {
  auto trace_christoffel =
      make_with_value<tnsr::i<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    trace_christoffel.get(i) =
        make_with_value<DataType>(used_for_size, 3. * i - 2.);
  }
  return trace_christoffel;
}

template <size_t SpatialDim, typename DataType>
tnsr::I<DataType, SpatialDim> make_trace_spatial_christoffel_second_kind(
    const DataType& used_for_size) {
  auto trace_christoffel =
      make_with_value<tnsr::I<DataType, SpatialDim>>(used_for_size, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    trace_christoffel.get(i) =
        make_with_value<DataType>(used_for_size, 3. * i - 2.5);
  }
  return trace_christoffel;
}


#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::I<DTYPE(data), DIM(data)> make_shift(const DTYPE(data) &    \
                                                      used_for_size);        \
  template tnsr::i<DTYPE(data), DIM(data)> make_lower_shift(                 \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::ii<DTYPE(data), DIM(data)> make_spatial_metric(             \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::II<DTYPE(data), DIM(data)> make_inverse_spatial_metric(     \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::A<DTYPE(data), DIM(data)> make_dummy_vector(                \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::a<DTYPE(data), DIM(data)> make_dummy_one_form(              \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::I<DTYPE(data), DIM(data)> make_dt_shift(const DTYPE(data) & \
                                                         used_for_size);     \
  template tnsr::ii<DTYPE(data), DIM(data)> make_dt_spatial_metric(          \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::aa<DTYPE(data), DIM(data)> make_dt_spacetime_metric(        \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::i<DTYPE(data), DIM(data)> make_deriv_lapse(                 \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::iJ<DTYPE(data), DIM(data)> make_deriv_shift(                \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::ijj<DTYPE(data), DIM(data)> make_deriv_spatial_metric(      \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::abb<DTYPE(data), DIM(data)>                                 \
  make_spacetime_deriv_spacetime_metric(const DTYPE(data) & used_for_size);  \
  template tnsr::iaa<DTYPE(data), DIM(data)>                                 \
  make_spatial_deriv_spacetime_metric(const DTYPE(data) & used_for_size);    \
  template tnsr::Abb<DTYPE(data), DIM(data)>                                 \
  make_spacetime_christoffel_second_kind(const DTYPE(data) & used_for_size); \
  template tnsr::Ijj<DTYPE(data), DIM(data)>                                 \
  make_spatial_christoffel_second_kind(const DTYPE(data) & used_for_size);   \
  template tnsr::iJkk<DTYPE(data), DIM(data)>                                \
  make_deriv_spatial_christoffel_second_kind(                                \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::aBcc<DTYPE(data), DIM(data)>                                \
  make_deriv_spacetime_christoffel_second_kind(                              \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::aa<DTYPE(data), DIM(data)> make_spacetime_metric(           \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::AA<DTYPE(data), DIM(data)> make_inverse_spacetime_metric(   \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::i<DTYPE(data), DIM(data)>                                   \
  make_trace_spatial_christoffel_first_kind(const DTYPE(data) &              \
                                            used_for_size);                  \
  template tnsr::I<DTYPE(data), DIM(data)>                                   \
  make_trace_spatial_christoffel_second_kind(const DTYPE(data) &             \
                                             used_for_size);

template Scalar<double> make_lapse(const double& used_for_size);
template Scalar<DataVector> make_lapse(const DataVector& used_for_size);
template Scalar<double> make_dt_lapse(const double& used_for_size);
template Scalar<DataVector> make_dt_lapse(const DataVector& used_for_size);
template Scalar<double> make_trace_extrinsic_curvature(
    const double& used_for_size);
template Scalar<DataVector> make_trace_extrinsic_curvature(
    const DataVector& used_for_size);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
