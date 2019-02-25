// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
template <size_t SpatialDim, typename Frame, typename DataType>
void phi(const gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> phi,
         const Scalar<DataType>& lapse,
         const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
         const tnsr::I<DataType, SpatialDim, Frame>& shift,
         const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
         const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
         const tnsr::ijj<DataType, SpatialDim, Frame>&
             deriv_spatial_metric) noexcept {
  if (UNLIKELY(get_size(get<0, 0, 0>(*phi)) != get_size(get(lapse)))) {
    *phi = tnsr::iaa<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  for (size_t k = 0; k < SpatialDim; ++k) {
    phi->get(k, 0, 0) = -2. * get(lapse) * deriv_lapse.get(k);
    for (size_t m = 0; m < SpatialDim; ++m) {
      for (size_t n = 0; n < SpatialDim; ++n) {
        phi->get(k, 0, 0) +=
            deriv_spatial_metric.get(k, m, n) * shift.get(m) * shift.get(n) +
            2. * spatial_metric.get(m, n) * shift.get(m) *
                deriv_shift.get(k, n);
      }
    }

    for (size_t i = 0; i < SpatialDim; ++i) {
      phi->get(k, 0, i + 1) = 0.;
      for (size_t m = 0; m < SpatialDim; ++m) {
        phi->get(k, 0, i + 1) +=
            deriv_spatial_metric.get(k, m, i) * shift.get(m) +
            spatial_metric.get(m, i) * deriv_shift.get(k, m);
      }
      for (size_t j = i; j < SpatialDim; ++j) {
        phi->get(k, i + 1, j + 1) = deriv_spatial_metric.get(k, i, j);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> phi(
    const Scalar<DataType>& lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept {
  tnsr::iaa<DataType, SpatialDim, Frame> var_phi{};
  GeneralizedHarmonic::phi<SpatialDim, Frame, DataType>(
      make_not_null(&var_phi), lapse, deriv_lapse, shift, deriv_shift,
      spatial_metric, deriv_spatial_metric);
  return var_phi;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void pi(const gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> pi,
        const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
        const tnsr::I<DataType, SpatialDim, Frame>& shift,
        const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
        const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
        const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
        const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*pi)) != get_size(get(lapse)))) {
    *pi = tnsr::aa<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }

  get<0, 0>(*pi) = -2. * get(lapse) * get(dt_lapse);

  for (size_t m = 0; m < SpatialDim; ++m) {
    for (size_t n = 0; n < SpatialDim; ++n) {
      get<0, 0>(*pi) +=
          dt_spatial_metric.get(m, n) * shift.get(m) * shift.get(n) +
          2. * spatial_metric.get(m, n) * shift.get(m) * dt_shift.get(n);
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    pi->get(0, i + 1) = 0.;
    for (size_t m = 0; m < SpatialDim; ++m) {
      pi->get(0, i + 1) += dt_spatial_metric.get(m, i) * shift.get(m) +
                           spatial_metric.get(m, i) * dt_shift.get(m);
    }
    for (size_t j = i; j < SpatialDim; ++j) {
      pi->get(i + 1, j + 1) = dt_spatial_metric.get(i, j);
    }
  }
  for (size_t mu = 0; mu < SpatialDim + 1; ++mu) {
    for (size_t nu = mu; nu < SpatialDim + 1; ++nu) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        pi->get(mu, nu) -= shift.get(i) * phi.get(i, mu, nu);
      }
      // Division by `lapse` here is somewhat more efficient (in Release mode)
      // than pre-computing `one_over_lapse` outside the loop for DataVectors
      // of `size` up to `50`. This is why we the next line is as it is.
      pi->get(mu, nu) /= -get(lapse);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> pi(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  tnsr::aa<DataType, SpatialDim, Frame> pi{};
  GeneralizedHarmonic::pi<SpatialDim, Frame, DataType>(
      make_not_null(&pi), lapse, dt_lapse, shift, dt_shift, spatial_metric,
      dt_spatial_metric, phi);
  return pi;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> gauge_source(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const Scalar<DataType>& trace_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>&
        trace_christoffel_last_indices) noexcept {
  DataType one_over_lapse = 1. / get(lapse);
  auto gauge_source_h =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame>>(lapse, 0.);

  // Temporary to avoid more nested loops.
  auto temp = dt_shift;
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      temp.get(i) -= shift.get(k) * deriv_shift.get(k, i);
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      gauge_source_h.get(i + 1) += spatial_metric.get(i, k) * temp.get(k);
    }
    gauge_source_h.get(i + 1) *= square(one_over_lapse);
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    gauge_source_h.get(i + 1) += one_over_lapse * deriv_lapse.get(i) -
                                 trace_christoffel_last_indices.get(i);
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    get<0>(gauge_source_h) +=
        shift.get(i) *
        (gauge_source_h.get(i + 1) + deriv_lapse.get(i) * one_over_lapse);
  }
  get<0>(gauge_source_h) -= one_over_lapse * get(dt_lapse) +
                            get(lapse) * get(trace_extrinsic_curvature);

  return gauge_source_h;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature(
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  auto ex_curv = make_with_value<tnsr::ii<DataType, SpatialDim, Frame>>(pi, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      for (size_t a = 0; a <= SpatialDim; ++a) {
        ex_curv.get(i, j) += 0.5 *
                             (phi.get(i, j + 1, a) + phi.get(j, i + 1, a)) *
                             spacetime_normal_vector.get(a);
      }
      ex_curv.get(i, j) += 0.5 * pi.get(i + 1, j + 1);
    }
  }
  return ex_curv;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void deriv_spatial_metric(
    const gsl::not_null<tnsr::ijj<DataType, SpatialDim, Frame>*>
        d_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  if (UNLIKELY(get_size(get<0, 0, 0>(*d_spatial_metric)) !=
               get_size(get<0, 0, 0>(phi)))) {
    *d_spatial_metric =
        tnsr::ijj<DataType, SpatialDim, Frame>(get_size(get<0, 0, 0>(phi)));
  }
  for (size_t k = 0; k < SpatialDim; ++k) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = i; j < SpatialDim; ++j) {
        d_spatial_metric->get(k, i, j) = phi.get(k, i + 1, j + 1);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ijj<DataType, SpatialDim, Frame> deriv_spatial_metric(
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  tnsr::ijj<DataType, SpatialDim, Frame> d_spatial_metric{};
  GeneralizedHarmonic::deriv_spatial_metric<SpatialDim, Frame, DataType>(
      make_not_null(&d_spatial_metric), phi);
  return d_spatial_metric;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_deriv_of_lapse(
    const gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*> deriv_lapse,
    const Scalar<DataType>& lapse,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  if (UNLIKELY(get_size(get<0>(*deriv_lapse)) != get_size(get(lapse)))) {
    *deriv_lapse = tnsr::i<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    deriv_lapse->get(i) =
        phi.get(i, 0, 0) * square(get<0>(spacetime_unit_normal));
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        if (LIKELY(a != 0 or b != 0)) {
          deriv_lapse->get(i) += phi.get(i, a, b) *
                                 spacetime_unit_normal.get(a) *
                                 spacetime_unit_normal.get(b);
        }
      }
    }
    deriv_lapse->get(i) *= -0.5 * get(lapse);
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::i<DataType, SpatialDim, Frame> spatial_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  tnsr::i<DataType, SpatialDim, Frame> deriv_lapse{};
  GeneralizedHarmonic::spatial_deriv_of_lapse<SpatialDim, Frame, DataType>(
      make_not_null(&deriv_lapse), lapse, spacetime_unit_normal, phi);
  return deriv_lapse;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_lapse(
    const gsl::not_null<Scalar<DataType>*> dt_lapse,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  if (UNLIKELY(get_size(get(*dt_lapse)) != get_size(get(lapse)))) {
    *dt_lapse = Scalar<DataType>(get_size(get(lapse)));
  }
  get(*dt_lapse) =
      get(lapse) * get<0, 0>(pi) * square(get<0>(spacetime_unit_normal));
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      // first term
      if (LIKELY(a != 0 or b != 0)) {
        get(*dt_lapse) += get(lapse) * pi.get(a, b) *
                          spacetime_unit_normal.get(a) *
                          spacetime_unit_normal.get(b);
      }
      // second term
      for (size_t i = 0; i < SpatialDim; ++i) {
        get(*dt_lapse) -= shift.get(i) * phi.get(i, a, b) *
                          spacetime_unit_normal.get(a) *
                          spacetime_unit_normal.get(b);
      }
    }
  }
  get(*dt_lapse) *= 0.5 * get(lapse);
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> time_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  Scalar<DataType> dt_lapse{};
  GeneralizedHarmonic::time_deriv_of_lapse<SpatialDim, Frame, DataType>(
      make_not_null(&dt_lapse), lapse, shift, spacetime_unit_normal, phi, pi);
  return dt_lapse;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_spatial_metric(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        dt_spatial_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*dt_spatial_metric)) !=
               get_size(get(lapse)))) {
    *dt_spatial_metric =
        tnsr::ii<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      dt_spatial_metric->get(i, j) = -get(lapse) * pi.get(i + 1, j + 1);
      for (size_t k = 0; k < SpatialDim; ++k) {
        dt_spatial_metric->get(i, j) += shift.get(k) * phi.get(k, i + 1, j + 1);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> time_deriv_of_spatial_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  tnsr::ii<DataType, SpatialDim, Frame> dt_spatial_metric{};
  GeneralizedHarmonic::time_deriv_of_spatial_metric<SpatialDim, Frame,
                                                    DataType>(
      make_not_null(&dt_spatial_metric), lapse, shift, phi, pi);
  return dt_spatial_metric;
}

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
struct D4gBuffer;

template <size_t SpatialDim, typename Frame>
struct D4gBuffer<SpatialDim, Frame, double> {
  explicit D4gBuffer(const size_t /*size*/) noexcept {}

  tnsr::ijj<double, SpatialDim, Frame> deriv_of_g{};
  Scalar<double> det_spatial_metric{};
};

template <size_t SpatialDim, typename Frame>
struct D4gBuffer<SpatialDim, Frame, DataVector> {
 private:
  // We make one giant allocation so that we don't thrash the heap.
  Variables<tmpl::list<::Tags::Tempijj<0, SpatialDim, Frame, DataVector>,
                       ::Tags::TempScalar<1, DataVector>>>
      buffer_;

 public:
  explicit D4gBuffer(const size_t size) noexcept
      : buffer_(size),
        deriv_of_g(
            get<::Tags::Tempijj<0, SpatialDim, Frame, DataVector>>(buffer_)),
        det_spatial_metric(get<::Tags::TempScalar<1, DataVector>>(buffer_)) {}

  tnsr::ijj<DataVector, SpatialDim, Frame>& deriv_of_g;
  Scalar<DataVector>& det_spatial_metric;
};
}  // namespace

template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_det_spatial_metric(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*>
        d4_det_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  if (UNLIKELY(get_size(get<0>(*d4_det_spatial_metric)) !=
               get_size(get(sqrt_det_spatial_metric)))) {
    *d4_det_spatial_metric = tnsr::a<DataType, SpatialDim, Frame>(
        get_size(get(sqrt_det_spatial_metric)));
  }
  auto& d4_g = *d4_det_spatial_metric;
  // Use a Variables to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  D4gBuffer<SpatialDim, Frame, DataType> buffer(
      get_size(get(sqrt_det_spatial_metric)));
  deriv_spatial_metric<SpatialDim, Frame, DataType>(
      make_not_null(&buffer.deriv_of_g), phi);
  get(buffer.det_spatial_metric) = square(get(sqrt_det_spatial_metric));
  // \f$ \partial_0 g = g g^{jk} \partial_0 g_{jk}\f$
  get<0>(d4_g) = inverse_spatial_metric.get(0, 0) * dt_spatial_metric.get(0, 0);
  for (size_t j = 0; j < SpatialDim; ++j) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      if (LIKELY(j != 0 or k != 0)) {
        get<0>(d4_g) +=
            inverse_spatial_metric.get(j, k) * dt_spatial_metric.get(j, k);
      }
    }
  }
  get<0>(d4_g) *= get(buffer.det_spatial_metric);
  // \f$ \partial_i g = g g^{jk} \partial_i g_{jk}\f$
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_g.get(i + 1) =
        inverse_spatial_metric.get(0, 0) * buffer.deriv_of_g.get(i, 0, 0);
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        if (LIKELY(j != 0 or k != 0)) {
          d4_g.get(i + 1) +=
              inverse_spatial_metric.get(j, k) * buffer.deriv_of_g.get(i, j, k);
        }
      }
    }
    d4_g.get(i + 1) *= get(buffer.det_spatial_metric);
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_det_spatial_metric(
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  tnsr::a<DataType, SpatialDim, Frame> d4_det_spatial_metric{};
  GeneralizedHarmonic::spacetime_deriv_of_det_spatial_metric<SpatialDim, Frame,
                                                             DataType>(
      make_not_null(&d4_det_spatial_metric), sqrt_det_spatial_metric,
      inverse_spatial_metric, dt_spatial_metric, phi);
  return d4_det_spatial_metric;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_deriv_of_shift(
    const gsl::not_null<tnsr::iJ<DataType, SpatialDim, Frame>*> deriv_shift,
    const Scalar<DataType>& lapse,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*deriv_shift)) != get_size(get(lapse)))) {
    *deriv_shift = tnsr::iJ<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      deriv_shift->get(i, j) =
          (inverse_spacetime_metric.get(j + 1, 0) +
           spacetime_unit_normal.get(j + 1) * spacetime_unit_normal.get(0)) *
          spacetime_unit_normal.get(0) * phi.get(i, 0, 0);
      for (size_t a = 0; a < SpatialDim + 1; ++a) {
        for (size_t b = 0; b < SpatialDim + 1; ++b) {
          if (a != 0 or b != 0) {
            deriv_shift->get(i, j) += (inverse_spacetime_metric.get(j + 1, a) +
                                       spacetime_unit_normal.get(j + 1) *
                                           spacetime_unit_normal.get(a)) *
                                      spacetime_unit_normal.get(b) *
                                      phi.get(i, a, b);
          }
        }
      }
      deriv_shift->get(i, j) *= get(lapse);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iJ<DataType, SpatialDim, Frame> spatial_deriv_of_shift(
    const Scalar<DataType>& lapse,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  tnsr::iJ<DataType, SpatialDim, Frame> deriv_shift{};
  GeneralizedHarmonic::spatial_deriv_of_shift<SpatialDim, Frame, DataType>(
      make_not_null(&deriv_shift), lapse, inverse_spacetime_metric,
      spacetime_unit_normal, phi);
  return deriv_shift;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_shift(
    const gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> dt_shift,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  if (UNLIKELY(get_size(get<0>(*dt_shift)) != get_size(get(lapse)))) {
    *dt_shift = tnsr::I<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    dt_shift->get(i) = -get(lapse) * pi.get(1, 0) *
                       spacetime_unit_normal.get(0) *
                       inverse_spatial_metric.get(i, 0);
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t a = 0; a < SpatialDim + 1; ++a) {
        if (a != 0 or j != 0) {
          dt_shift->get(i) -= get(lapse) * pi.get(j + 1, a) *
                              spacetime_unit_normal.get(a) *
                              inverse_spatial_metric.get(i, j);
        }
        for (size_t k = 0; k < SpatialDim; ++k) {
          dt_shift->get(i) += shift.get(j) * spacetime_unit_normal.get(a) *
                              phi.get(j, k + 1, a) *
                              inverse_spatial_metric.get(i, k);
        }
      }
    }
    dt_shift->get(i) *= get(lapse);
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::I<DataType, SpatialDim, Frame> time_deriv_of_shift(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  tnsr::I<DataType, SpatialDim, Frame> dt_shift{};
  GeneralizedHarmonic::time_deriv_of_shift<SpatialDim, Frame, DataType>(
      make_not_null(&dt_shift), lapse, shift, inverse_spatial_metric,
      spacetime_unit_normal, phi, pi);
  return dt_shift;
}

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
struct D0LowerShiftBuffer;

template <size_t SpatialDim, typename Frame>
struct D0LowerShiftBuffer<SpatialDim, Frame, double> {
  explicit D0LowerShiftBuffer(const size_t /*size*/) noexcept {}

  tnsr::I<double, SpatialDim, Frame> dt_shift{};
  tnsr::ii<double, SpatialDim, Frame> dt_spatial_metric{};
};

template <size_t SpatialDim, typename Frame>
struct D0LowerShiftBuffer<SpatialDim, Frame, DataVector> {
 private:
  // We make one giant allocation so that we don't thrash the heap.
  Variables<tmpl::list<::Tags::TempI<0, SpatialDim, Frame, DataVector>,
                       ::Tags::Tempii<1, SpatialDim, Frame, DataVector>>>
      buffer_;

 public:
  explicit D0LowerShiftBuffer(const size_t size) noexcept
      : buffer_(size),
        dt_shift(get<::Tags::TempI<0, SpatialDim, Frame, DataVector>>(buffer_)),
        dt_spatial_metric(
            get<::Tags::Tempii<1, SpatialDim, Frame, DataVector>>(buffer_)) {}

  tnsr::I<DataVector, SpatialDim, Frame>& dt_shift;
  tnsr::ii<DataVector, SpatialDim, Frame>& dt_spatial_metric;
};
}  // namespace

template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_lower_shift(
    const gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*> dt_lower_shift,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  if (UNLIKELY(get_size(get<0>(*dt_lower_shift)) != get_size(get(lapse)))) {
    *dt_lower_shift =
        tnsr::i<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  // Use a Variables to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  D0LowerShiftBuffer<SpatialDim, Frame, DataType> buffer(get_size(get(lapse)));
  // get \partial_0 N^j
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  GeneralizedHarmonic::time_deriv_of_shift<SpatialDim, Frame, DataType>(
      make_not_null(&buffer.dt_shift), lapse, shift, inverse_spatial_metric,
      spacetime_unit_normal, phi, pi);
  GeneralizedHarmonic::time_deriv_of_spatial_metric<SpatialDim, Frame,
                                                    DataType>(
      make_not_null(&buffer.dt_spatial_metric), lapse, shift, phi, pi);
  for (size_t i = 0; i < SpatialDim; ++i) {
    dt_lower_shift->get(i) = spatial_metric.get(i, 0) * buffer.dt_shift.get(0) +
                             shift.get(0) * buffer.dt_spatial_metric.get(i, 0);
    for (size_t j = 0; j < SpatialDim; ++j) {
      if (j != 0) {
        dt_lower_shift->get(i) +=
            spatial_metric.get(i, j) * buffer.dt_shift.get(j) +
            shift.get(j) * buffer.dt_spatial_metric.get(i, j);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::i<DataType, SpatialDim, Frame> time_deriv_of_lower_shift(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  tnsr::i<DataType, SpatialDim, Frame> dt_lower_shift{};
  GeneralizedHarmonic::time_deriv_of_lower_shift<SpatialDim, Frame, DataType>(
      make_not_null(&dt_lower_shift), lapse, shift, spatial_metric,
      spacetime_unit_normal, phi, pi);
  return dt_lower_shift;
}

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

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template void GeneralizedHarmonic::phi(                                     \
      const gsl::not_null<tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>*>    \
          var_phi,                                                            \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                   \
          deriv_spatial_metric) noexcept;                                     \
  template tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>                     \
  GeneralizedHarmonic::phi(                                                   \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                   \
          deriv_spatial_metric) noexcept;                                     \
  template void GeneralizedHarmonic::pi(                                      \
      const gsl::not_null<tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>*>     \
          var_pi,                                                             \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>                      \
  GeneralizedHarmonic::pi(                                                    \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                      \
  GeneralizedHarmonic::extrinsic_curvature(                                   \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_normal_vector,                                            \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template void GeneralizedHarmonic::deriv_spatial_metric(                    \
      const gsl::not_null<tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>*>    \
          d_spatial_metric,                                                   \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>                     \
  GeneralizedHarmonic::deriv_spatial_metric(                                  \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template void GeneralizedHarmonic::time_deriv_of_spatial_metric(            \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*>     \
          dt_spatial_metric,                                                  \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                      \
  GeneralizedHarmonic::time_deriv_of_spatial_metric(                          \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template void GeneralizedHarmonic::spacetime_deriv_of_det_spatial_metric(   \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>      \
          d4_det_spatial_metric,                                              \
      const Scalar<DTYPE(data)>& det_spatial_metric,                          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                       \
  GeneralizedHarmonic::spacetime_deriv_of_det_spatial_metric(                 \
      const Scalar<DTYPE(data)>& det_spatial_metric,                          \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template void GeneralizedHarmonic::spatial_deriv_of_lapse(                  \
      const gsl::not_null<tnsr::i<DTYPE(data), DIM(data), FRAME(data)>*>      \
          deriv_lapse,                                                        \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::i<DTYPE(data), DIM(data), FRAME(data)>                       \
  GeneralizedHarmonic::spatial_deriv_of_lapse(                                \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template void GeneralizedHarmonic::time_deriv_of_lapse(                     \
      const gsl::not_null<Scalar<DTYPE(data)>*> dt_lapse,                     \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template Scalar<DTYPE(data)> GeneralizedHarmonic::time_deriv_of_lapse(      \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template void GeneralizedHarmonic::spatial_deriv_of_shift(                  \
      const gsl::not_null<tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>*>     \
          deriv_shift,                                                        \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>                      \
  GeneralizedHarmonic::spatial_deriv_of_shift(                                \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template void GeneralizedHarmonic::time_deriv_of_shift(                     \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*>      \
          dt_shift,                                                           \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)>                       \
  GeneralizedHarmonic::time_deriv_of_shift(                                   \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template void GeneralizedHarmonic::time_deriv_of_lower_shift(               \
      const gsl::not_null<tnsr::i<DTYPE(data), DIM(data), FRAME(data)>*>      \
          dt_lower_shift,                                                     \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template tnsr::i<DTYPE(data), DIM(data), FRAME(data)>                       \
  GeneralizedHarmonic::time_deriv_of_lower_shift(                             \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template void GeneralizedHarmonic::spacetime_deriv_of_norm_of_shift(        \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>      \
          d4_norm_of_shift,                                                   \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                       \
  GeneralizedHarmonic::spacetime_deriv_of_norm_of_shift(                      \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spacetime_metric,                                           \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi) noexcept;      \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                       \
  GeneralizedHarmonic::gauge_source(                                          \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,  \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,           \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const Scalar<DTYPE(data)>& trace_extrinsic_curvature,                   \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&                     \
          trace_christoffel_last_indices) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
