// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedWaveHelpers.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/HalfPiPhiTwoNormals.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace gh::gauges {
namespace DampedHarmonicGauge_detail {
// Roll-on function for the damped harmonic gauge.
//
// For times after \f$t_0\f$, compute the roll-on function
// \f$ R(t) = 1 - \exp(-((t - t_0)/\sigma_t)^4]) \f$,
// and return \f$ R(t) = 0\f$ at times before.
double roll_on_function(const double time, const double t_start,
                        const double sigma_t) {
  if (time < t_start) {
    return 0.;
  }
  return 1. - exp(-pow<4>((time - t_start) / sigma_t));
}

// Time derivative of the damped harmonic gauge roll-on function.
//
// \details Compute the time derivative:
// \f{align*}
// \partial_0 R(t) = 4 exp[-((t - t0)/\sigma_t)^4] (t - t0)^3 / sigma_t^4
double time_deriv_of_roll_on_function(const double time, const double t_start,
                                      const double sigma_t) {
  if (time < t_start) {
    return 0.;
  }
  const double time_since_start_over_sigma = (time - t_start) / sigma_t;
  return exp(-pow<4>(time_since_start_over_sigma)) * 4. *
         pow<3>(time_since_start_over_sigma) / sigma_t;
}
}  // namespace DampedHarmonicGauge_detail

namespace {
template <bool UseRollon, size_t SpatialDim, typename Frame>
void damped_harmonic_impl(
    const gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame>*> gauge_h,
    const gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame>*> d4_gauge_h,
    const tnsr::a<DataVector, SpatialDim, Frame>* gauge_h_init,
    const tnsr::ab<DataVector, SpatialDim, Frame>* dgauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const tnsr::a<DataVector, SpatialDim, Frame>&
        spacetime_unit_normal_one_form,
    const tnsr::A<DataVector, SpatialDim, Frame>& spacetime_unit_normal,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::abb<DataVector, SpatialDim, Frame>& d4_spacetime_metric,
    const Scalar<DataVector>& half_pi_two_normals,
    const tnsr::i<DataVector, SpatialDim, Frame>& half_phi_two_normals,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, const double time,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double amp_coef_L1, const double amp_coef_L2, const double amp_coef_S,
    const int exp_L1, const int exp_L2, const int exp_S,
    const double rollon_start_time, const double rollon_width,
    const double sigma_r) {
  const size_t num_points = get(lapse).size();
  destructive_resize_components(gauge_h, num_points);
  destructive_resize_components(d4_gauge_h, num_points);

  if constexpr (UseRollon) {
    ASSERT(gauge_h_init != nullptr,
           "Cannot call damped_harmonic_impl with UseRollon enabled and "
           "gauge_h_init being nullptr");
    ASSERT(dgauge_h_init != nullptr,
           "Cannot call damped_harmonic_impl with UseRollon enabled and "
           "dgauge_h_init being nullptr");
  } else {
    ASSERT(gauge_h_init == nullptr,
           "Cannot call damped_harmonic_impl with UseRollon disabled and "
           "gauge_h_init not being nullptr");
    ASSERT(dgauge_h_init == nullptr,
           "Cannot call damped_harmonic_impl with UseRollon disabled and "
           "dgauge_h_init not being nullptr");
  }

  // Use a TempBuffer to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  TempBuffer<tmpl::list<
      ::Tags::Tempii<1, SpatialDim, Frame>,
      ::Tags::Tempijj<4, SpatialDim, Frame>, ::Tags::TempScalar<5>,
      ::Tags::TempScalar<6>, ::Tags::TempScalar<7>, ::Tags::TempScalar<8>,
      ::Tags::TempScalar<9>, ::Tags::TempScalar<10>, ::Tags::TempScalar<11>,
      ::Tags::TempScalar<12>, ::Tags::TempScalar<13>, ::Tags::TempScalar<14>,
      ::Tags::Tempa<15, SpatialDim, Frame>,
      ::Tags::Tempa<16, SpatialDim, Frame>,
      ::Tags::Tempa<17, SpatialDim, Frame>,
      ::Tags::Tempa<18, SpatialDim, Frame>,
      ::Tags::Tempa<19, SpatialDim, Frame>,
      ::Tags::Tempa<20, SpatialDim, Frame>,
      ::Tags::Tempa<21, SpatialDim, Frame>,
      ::Tags::Tempa<22, SpatialDim, Frame>,
      ::Tags::Tempa<23, SpatialDim, Frame>,
      ::Tags::Tempa<24, SpatialDim, Frame>,
      ::Tags::Tempa<27, SpatialDim, Frame>,
      ::Tags::TempaB<30, SpatialDim, Frame>,
      ::Tags::Tempa<31, SpatialDim, Frame>, ::Tags::TempScalar<32>,
      ::Tags::Tempa<33, SpatialDim, Frame>,
      ::Tags::Tempab<35, SpatialDim, Frame>,
      ::Tags::Tempa<36, SpatialDim, Frame>,
      ::Tags::Tempab<37, SpatialDim, Frame>>>
      buffer(num_points);
  auto& one_over_lapse = get<::Tags::TempScalar<5>>(buffer);
  auto& log_fac_1 = get<::Tags::TempScalar<6>>(buffer);
  auto& log_fac_2 = get<::Tags::TempScalar<7>>(buffer);
  auto& weight = get<::Tags::TempScalar<8>>(buffer);
  auto& mu_L1 = get<::Tags::TempScalar<9>>(buffer);
  auto& mu_S = get<::Tags::TempScalar<10>>(buffer);
  auto& mu_L2 = get<::Tags::TempScalar<11>>(buffer);
  auto& mu_S_over_lapse = get<::Tags::TempScalar<12>>(buffer);
  auto& mu1 = get<::Tags::TempScalar<13>>(buffer);
  auto& mu2 = get<::Tags::TempScalar<14>>(buffer);
  auto& d4_weight = get<::Tags::Tempa<15, SpatialDim, Frame>>(buffer);
  auto& d4_RW_L1 = get<::Tags::Tempa<16, SpatialDim, Frame>>(buffer);
  auto& d4_RW_S = get<::Tags::Tempa<17, SpatialDim, Frame>>(buffer);
  auto& d4_RW_L2 = get<::Tags::Tempa<18, SpatialDim, Frame>>(buffer);
  auto& d4_mu_S = get<::Tags::Tempa<19, SpatialDim, Frame>>(buffer);
  auto& d4_mu1 = get<::Tags::Tempa<20, SpatialDim, Frame>>(buffer);
  auto& d4_mu2 = get<::Tags::Tempa<21, SpatialDim, Frame>>(buffer);
  auto& d4_log_fac_mu1 = get<::Tags::Tempa<22, SpatialDim, Frame>>(buffer);
  auto& d4_log_fac_muS = get<::Tags::Tempa<23, SpatialDim, Frame>>(buffer);
  auto& d4_log_fac_mu2 = get<::Tags::Tempa<24, SpatialDim, Frame>>(buffer);

  auto& spacetime_metric_dot_shift =
      get<::Tags::Tempa<31, SpatialDim, Frame>>(buffer);
  auto& prefac = get<::Tags::TempScalar<32>>(buffer);
  auto& d4_muS_over_lapse = get<::Tags::Tempa<33, SpatialDim, Frame>>(buffer);
  auto& dT1 = get<::Tags::Tempab<35, SpatialDim, Frame>>(buffer);
  auto& dT2 = get<::Tags::Tempa<36, SpatialDim, Frame>>(buffer);
  auto& dT3 = get<::Tags::Tempab<37, SpatialDim, Frame>>(buffer);

  // 3+1 quantities
  const tnsr::ii<DataVector, SpatialDim, Frame> spatial_metric{};
  for (size_t i = 0; i< SpatialDim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      make_const_view(make_not_null(&spatial_metric.get(i, j)),
                      spacetime_metric.get(i + 1, j + 1), 0, num_points);
    }
  }
  // We need \f$ \partial_a g_{bi} = \partial_a \psi_{bi} \f$. Here we
  // use `derivatives_of_spacetime_metric` to get \f$ \partial_a g_{bc}\f$
  // instead, and use only the derivatives of \f$ g_{bi}\f$.
  const tnsr::ii<DataVector, SpatialDim, Frame> d0_spatial_metric{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      make_const_view(make_not_null(&d0_spatial_metric.get(i, j)),
                      d4_spacetime_metric.get(0, i + 1, j + 1), 0, num_points);
    }
  }

  // commonly used terms
  constexpr auto exp_fac_1 = 0.5;
  constexpr auto exp_fac_2 = 0.;
  get(one_over_lapse) = 1. / get(lapse);
  DampedHarmonicGauge_detail::log_factor_metric_lapse<DataVector>(
      make_not_null(&log_fac_1), lapse, sqrt_det_spatial_metric, exp_fac_1);
  DampedHarmonicGauge_detail::log_factor_metric_lapse<DataVector>(
      make_not_null(&log_fac_2), lapse, sqrt_det_spatial_metric, exp_fac_2);

  // Tempering functions
  const double roll_on = UseRollon
                             ? DampedHarmonicGauge_detail::roll_on_function(
                                   time, rollon_start_time, rollon_width)
                             : 1.0;
  DampedHarmonicGauge_detail::spatial_weight_function<DataVector, SpatialDim,
                                                      Frame>(
      make_not_null(&weight), coords, sigma_r);

  // coeffs that enter gauge source function
  get(mu_L1) =
      amp_coef_L1 * roll_on * get(weight) * pow(get(log_fac_1), exp_L1);
  get(mu_S) = amp_coef_S * roll_on * get(weight) * pow(get(log_fac_1), exp_S);
  get(mu_L2) =
      amp_coef_L2 * roll_on * get(weight) * pow(get(log_fac_2), exp_L2);
  get(mu_S_over_lapse) = get(mu_S) * get(one_over_lapse);

  // Calc \f$ \mu_1 = \mu_{L1} log(rootg/N) = R W log(rootg/N)^5\f$
  get(mu1) = get(mu_L1) * get(log_fac_1);

  // Calc \f$ \mu_2 = \mu_{L2} log(1/N) = R W log(1/N)^5\f$
  get(mu2) = get(mu_L2) * get(log_fac_2);

  get(prefac) = get(mu_L1) * get(log_fac_1) + get(mu_L2) * get(log_fac_2);

  // Compute g_ai shift^i
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    spacetime_metric_dot_shift.get(a) =
        spacetime_metric.get(a, 1) * shift.get(0);
    for (size_t i = 1; i < SpatialDim; ++i) {
      spacetime_metric_dot_shift.get(a) +=
          spacetime_metric.get(a, i + 1) * shift.get(i);
    }
  }

  // Calculate H_a
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    gauge_h->get(a) = -get(mu_S_over_lapse) * spacetime_metric_dot_shift.get(a);
    if constexpr (UseRollon) {
      gauge_h->get(a) += (1. - roll_on) * gauge_h_init->get(a);
    }
  }
  // Since n_i = 0 only do H_0 term
  gauge_h->get(0) += get(prefac) * spacetime_unit_normal_one_form.get(0);

  [[maybe_unused]] const double d0_roll_on =
      UseRollon ? DampedHarmonicGauge_detail::time_deriv_of_roll_on_function(
                      time, rollon_start_time, rollon_width)
                : 0.0;

  // Calc \f$ \partial_a [R W] \f$
  DampedHarmonicGauge_detail::spacetime_deriv_of_spatial_weight_function<
      DataVector, SpatialDim, Frame>(make_not_null(&d4_weight), coords, sigma_r,
                                     weight);
  d4_RW_L1 = d4_weight;
  d4_RW_S = d4_weight;
  d4_RW_L2 = d4_weight;
  if constexpr (UseRollon) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_RW_L1.get(a) *= roll_on;
      d4_RW_S.get(a) *= roll_on;
      d4_RW_L2.get(a) *= roll_on;
    }
    get<0>(d4_RW_L1) += get(weight) * d0_roll_on;
    get<0>(d4_RW_S) += get(weight) * d0_roll_on;
    get<0>(d4_RW_L2) += get(weight) * d0_roll_on;
  }

  // \partial_a \mu_{S} = \partial_a(A_S R_S W
  //                               \log(\sqrt{g}/N)^{c_{S}})
  // \partial_a \mu_1 = \partial_a(A_L1 R_L1 W
  //                               \log(\sqrt{g}/N)^{1+c_{L1}})
  // \partial_a \mu_2 = \partial_a(A_L2 R_L2 W
  //                               \log(1/N)^{1+c_{L2}})
  DampedHarmonicGauge_detail::spacetime_deriv_of_power_log_factor_metric_lapse<
      DataVector, SpatialDim, Frame>(
      make_not_null(&d4_log_fac_mu1), lapse, shift, spacetime_unit_normal,
      inverse_spatial_metric, sqrt_det_spatial_metric, d0_spatial_metric, pi,
      phi, exp_fac_1, exp_L1 + 1);
  DampedHarmonicGauge_detail::spacetime_deriv_of_power_log_factor_metric_lapse<
      DataVector, SpatialDim, Frame>(
      make_not_null(&d4_log_fac_muS), lapse, shift, spacetime_unit_normal,
      inverse_spatial_metric, sqrt_det_spatial_metric, d0_spatial_metric, pi,
      phi, exp_fac_1, exp_S);
  DampedHarmonicGauge_detail::spacetime_deriv_of_power_log_factor_metric_lapse<
      DataVector, SpatialDim, Frame>(
      make_not_null(&d4_log_fac_mu2), lapse, shift, spacetime_unit_normal,
      inverse_spatial_metric, sqrt_det_spatial_metric, d0_spatial_metric, pi,
      phi, exp_fac_2, exp_L2 + 1);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    // \f$ \partial_a \mu_1 \f$
    d4_mu1.get(a) =
        amp_coef_L1 * pow(get(log_fac_1), exp_L1 + 1) * d4_RW_L1.get(a) +
        amp_coef_L1 * roll_on * get(weight) * d4_log_fac_mu1.get(a);
    // \f$ \partial_a \mu_{S} \f$
    d4_mu_S.get(a) = amp_coef_S * d4_RW_S.get(a) * pow(get(log_fac_1), exp_S) +
                     amp_coef_S * roll_on * get(weight) * d4_log_fac_muS.get(a);
    // \f$ \partial_a \mu_2 \f$
    d4_mu2.get(a) =
        amp_coef_L2 * pow(get(log_fac_2), exp_L2 + 1) * d4_RW_L2.get(a) +
        amp_coef_L2 * roll_on * get(weight) * d4_log_fac_mu2.get(a);
  }

  const tnsr::ijj<DataVector, SpatialDim, Frame> d3_spatial_metric{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k <= j; ++k) {
        make_const_view(make_not_null(&d3_spatial_metric.get(i, j, k)),
                        phi.get(i, j + 1, k + 1), 0, num_points);
      }
    }
  }

  // Calc \f$ \partial_a T1 \f$
  if constexpr (UseRollon) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      dT1.get(0, b) = -gauge_h_init->get(b) * d0_roll_on;
      for (size_t a = 0; a < SpatialDim + 1; ++a) {
        if (a != 0) {
          dT1.get(a, b) = (1. - roll_on) * dgauge_h_init->get(a, b);
        } else {
          dT1.get(a, b) += (1. - roll_on) * dgauge_h_init->get(a, b);
        }
      }
    }
  }

  // d_t lapse / lapse = 0.5 * n^a n^b (lapse Pi_{ab} - shift^i Phi_{iab})
  get<0>(d4_muS_over_lapse) = -get<0>(shift) * get<0>(half_phi_two_normals);
  for (size_t i = 1; i < SpatialDim; ++i) {
    get<0>(d4_muS_over_lapse) -= shift.get(i) * half_phi_two_normals.get(i);
  }
  get<0>(d4_muS_over_lapse) += get(lapse) * get(half_pi_two_normals);
  get<0>(d4_muS_over_lapse) *= get(lapse);

  // Calc \f$ \partial_a T2 \f$
  get<0>(dT2) =
      (get<0>(d4_mu1) + get<0>(d4_mu2)) * get<0>(spacetime_unit_normal_one_form)
      // Note:  \f$ \partial_a n_b = {-\partial_a lapse, 0, 0, 0} \f$
      - (get(mu1) + get(mu2)) * get<0>(d4_muS_over_lapse);

  for (size_t i = 0; i < SpatialDim; ++i) {
    dT2.get(i + 1) =
        (d4_mu1.get(i + 1) + d4_mu2.get(i + 1)) *
            get<0>(spacetime_unit_normal_one_form)
        // Note:  \f$ \partial_a n_b = {-\partial_a lapse, 0, 0, 0} \f$
        + (get(mu1) + get(mu2)) * get(lapse) * half_phi_two_normals.get(i);
  }

  // \f[ \partial_a (\mu_S/N) = (1/N) \partial_a \mu_{S}
  //         - (\mu_{S}/N^2) \partial_a N
  // \f]
  //
  // Note that the d4_lapse terms are actually
  // d_t lapse / lapse = 0.5 * n^a n^b (lapse Pi_{ab} - shift^i Phi_{iab})
  // d_i lapse / lapse = -0.5 * n^a n^b Phi_{iab}
  //
  // The GH RHS computes:
  //  0.5 * n^a n^b Pi_{ab}
  //  0.5 * n^a n^b Phi_{iab}
  // so we reuse that work by taking them as arguments.
  get<0>(d4_muS_over_lapse) *= -get(mu_S);
  get<0>(d4_muS_over_lapse) += get<0>(d4_mu_S);
  get<0>(d4_muS_over_lapse) *= get(one_over_lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_muS_over_lapse.get(i + 1) =
        get(one_over_lapse) *
        (d4_mu_S.get(i + 1) + get(mu_S) * half_phi_two_normals.get(i));
  }

  // Calc \f$ \partial_a T3 \f$ (note minus sign)
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      dT3.get(a, j + 1) = d4_spacetime_metric.get(a, 0, j + 1);
    }
    dT3.get(a, 0) = d4_spacetime_metric.get(a, 0, 1) * get<0>(shift);
    for (size_t j = 1; j < SpatialDim; ++j) {
      dT3.get(a, 0) += d4_spacetime_metric.get(a, 0, j + 1) * shift.get(j);
    }
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = i + 1; j < SpatialDim; ++j) {
        dT3.get(a, 0) -= shift.get(i) * shift.get(j) *
                         d4_spacetime_metric.get(a, i + 1, j + 1);
      }
    }
    dT3.get(a, 0) *= 2.0;
    for (size_t i = 0; i < SpatialDim; ++i) {
      dT3.get(a, 0) -= shift.get(i) * shift.get(i) *
                       d4_spacetime_metric.get(a, i + 1, i + 1);
    }

    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      dT3.get(a, b) *= -get(mu_S_over_lapse);
      dT3.get(a, b) -=
          d4_muS_over_lapse.get(a) * spacetime_metric_dot_shift.get(b);
    }
  }

  // Calc \f$ \partial_a H_b = dT1_{ab} + dT2_{ab} + dT3_{ab} \f$
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      if constexpr (UseRollon) {
        d4_gauge_h->get(a, b) = dT1.get(a, b) +  dT3.get(a, b);
      } else {
        d4_gauge_h->get(a, b) = dT3.get(a, b);
      }
    }
    d4_gauge_h->get(a, 0) += dT2.get(a);
  }
}
}  // namespace

template <size_t SpatialDim, typename Frame>
void damped_harmonic_rollon(
    const gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame>*> gauge_h,
    const gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame>*> d4_gauge_h,
    const tnsr::a<DataVector, SpatialDim, Frame>& gauge_h_init,
    const tnsr::ab<DataVector, SpatialDim, Frame>& dgauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const tnsr::a<DataVector, SpatialDim, Frame>&
        spacetime_unit_normal_one_form,
    const tnsr::A<DataVector, SpatialDim, Frame>& spacetime_unit_normal,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::abb<DataVector, SpatialDim, Frame>& d4_spacetime_metric,
    const Scalar<DataVector>& half_pi_two_normals,
    const tnsr::i<DataVector, SpatialDim, Frame>& half_phi_two_normals,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, const double time,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double amp_coef_L1, const double amp_coef_L2, const double amp_coef_S,
    const int exp_L1, const int exp_L2, const int exp_S,
    const double rollon_start_time, const double rollon_width,
    const double sigma_r) {
  damped_harmonic_impl<true>(
      gauge_h, d4_gauge_h, &gauge_h_init, &dgauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, spacetime_unit_normal,
      sqrt_det_spatial_metric, inverse_spatial_metric, d4_spacetime_metric,
      half_pi_two_normals, half_phi_two_normals, spacetime_metric, pi, phi,
      time, coords, amp_coef_L1, amp_coef_L2, amp_coef_S, exp_L1, exp_L2, exp_S,
      rollon_start_time, rollon_width, sigma_r);
}

template <size_t SpatialDim, typename Frame>
void damped_harmonic(
    const gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame>*> gauge_h,
    const gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame>*> d4_gauge_h,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const tnsr::a<DataVector, SpatialDim, Frame>&
        spacetime_unit_normal_one_form,
    const tnsr::A<DataVector, SpatialDim, Frame>& spacetime_unit_normal,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::abb<DataVector, SpatialDim, Frame>& d4_spacetime_metric,
    const Scalar<DataVector>& half_pi_two_normals,
    const tnsr::i<DataVector, SpatialDim, Frame>& half_phi_two_normals,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double amp_coef_L1, const double amp_coef_L2, const double amp_coef_S,
    const int exp_L1, const int exp_L2, const int exp_S, const double sigma_r) {
  damped_harmonic_impl<false, SpatialDim, Frame>(
      gauge_h, d4_gauge_h, nullptr, nullptr, lapse, shift,
      spacetime_unit_normal_one_form, spacetime_unit_normal,
      sqrt_det_spatial_metric, inverse_spatial_metric, d4_spacetime_metric,
      half_pi_two_normals, half_phi_two_normals, spacetime_metric, pi, phi,
      std::numeric_limits<double>::signaling_NaN(), coords, amp_coef_L1,
      amp_coef_L2, amp_coef_S, exp_L1, exp_L2, exp_S,
      std::numeric_limits<double>::signaling_NaN(),
      std::numeric_limits<double>::signaling_NaN(), sigma_r);
}

DampedHarmonic::DampedHarmonic(const double width,
                               const std::array<double, 3>& amps,
                               const std::array<int, 3>& exps)
    : spatial_decay_width_(width), amplitudes_(amps), exponents_(exps) {}

DampedHarmonic::DampedHarmonic(CkMigrateMessage* const msg)
    : GaugeCondition(msg) {}

void DampedHarmonic::pup(PUP::er& p) {
  GaugeCondition::pup(p);
  p | spatial_decay_width_;
  p | amplitudes_;
  p | exponents_;
}

std::unique_ptr<GaugeCondition> DampedHarmonic::get_clone() const {
  return std::make_unique<DampedHarmonic>(*this);
}

template <size_t SpatialDim>
void DampedHarmonic::gauge_and_spacetime_derivative(
    const gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame::Inertial>*>
        gauge_h,
    const gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>*>
        d4_gauge_h,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& shift,
    const tnsr::a<DataVector, SpatialDim, Frame::Inertial>&
        spacetime_unit_normal_one_form,
    const tnsr::A<DataVector, SpatialDim, Frame::Inertial>&
        spacetime_unit_normal,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame::Inertial>&
        inverse_spatial_metric,
    const tnsr::abb<DataVector, SpatialDim, Frame::Inertial>&
        d4_spacetime_metric,
    const Scalar<DataVector>& half_pi_two_normals,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&
        half_phi_two_normals,
    const tnsr::aa<DataVector, SpatialDim, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame::Inertial>& phi,
    const double /*time*/,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& inertial_coords)
    const {
  damped_harmonic(
      gauge_h, d4_gauge_h, lapse, shift, spacetime_unit_normal_one_form,
      spacetime_unit_normal, sqrt_det_spatial_metric, inverse_spatial_metric,
      d4_spacetime_metric, half_pi_two_normals, half_phi_two_normals,
      spacetime_metric, pi, phi, inertial_coords, amplitudes_[0],
      amplitudes_[1], amplitudes_[2], exponents_[0], exponents_[1],
      exponents_[2], spatial_decay_width_);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID DampedHarmonic::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void DampedHarmonic::gauge_and_spacetime_derivative(                \
      gsl::not_null<tnsr::a<DataVector, DIM(data), Frame::Inertial>*> gauge_h, \
      gsl::not_null<tnsr::ab<DataVector, DIM(data), Frame::Inertial>*>         \
          d4_gauge_h,                                                          \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift,            \
      const tnsr::a<DataVector, DIM(data), Frame::Inertial>&                   \
          spacetime_unit_normal_one_form,                                      \
      const tnsr::A<DataVector, DIM(data), Frame::Inertial>&                   \
          spacetime_unit_normal,                                               \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                       \
      const tnsr::II<DataVector, DIM(data), Frame::Inertial>&                  \
          inverse_spatial_metric,                                              \
      const tnsr::abb<DataVector, DIM(data), Frame::Inertial>&                 \
          d4_spacetime_metric,                                                 \
      const Scalar<DataVector>& half_pi_two_normals,                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          half_phi_two_normals,                                                \
      const tnsr::aa<DataVector, DIM(data), Frame::Inertial>&                  \
          spacetime_metric,                                                    \
      const tnsr::aa<DataVector, DIM(data), Frame::Inertial>& pi,              \
      const tnsr::iaa<DataVector, DIM(data), Frame::Inertial>& phi,            \
      const double /*time*/,                                                   \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& inertial_coords)  \
      const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define DTYPE_SCAL(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_DV_FUNC(_, data)                                           \
  template void damped_harmonic_rollon(                                        \
      gsl::not_null<tnsr::a<DataVector, DIM(data), FRAME(data)>*> gauge_h,     \
      gsl::not_null<tnsr::ab<DataVector, DIM(data), FRAME(data)>*> d4_gauge_h, \
      const tnsr::a<DataVector, DIM(data), FRAME(data)>& gauge_h_init,         \
      const tnsr::ab<DataVector, DIM(data), FRAME(data)>& dgauge_h_init,       \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const tnsr::a<DataVector, DIM(data), FRAME(data)>&                       \
          spacetime_unit_normal_one_form,                                      \
      const tnsr::A<DataVector, DIM(data), FRAME(data)>&                       \
          spacetime_unit_normal,                                               \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                       \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const tnsr::abb<DataVector, DIM(data), FRAME(data)>&                     \
          d4_spacetime_metric,                                                 \
      const Scalar<DataVector>& half_pi_two_normals,                           \
      const tnsr::i<DataVector, DIM(data), FRAME(data)>& half_phi_two_normals, \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& spacetime_metric,    \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& pi,                  \
      const tnsr::iaa<DataVector, DIM(data), FRAME(data)>& phi,                \
      const double time,                                                       \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& coords,               \
      const double amp_coef_L1, const double amp_coef_L2,                      \
      const double amp_coef_S, const int exp_L1, const int exp_L2,             \
      const int exp_S, const double rollon_start_time,                         \
      const double rollon_width, const double sigma_r);                        \
  template void damped_harmonic(                                               \
      gsl::not_null<tnsr::a<DataVector, DIM(data), FRAME(data)>*> gauge_h,     \
      gsl::not_null<tnsr::ab<DataVector, DIM(data), FRAME(data)>*> d4_gauge_h, \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const tnsr::a<DataVector, DIM(data), FRAME(data)>&                       \
          spacetime_unit_normal_one_form,                                      \
      const tnsr::A<DataVector, DIM(data), FRAME(data)>&                       \
          spacetime_unit_normal,                                               \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                       \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const tnsr::abb<DataVector, DIM(data), FRAME(data)>&                     \
          d4_spacetime_metric,                                                 \
      const Scalar<DataVector>& half_pi_two_normals,                           \
      const tnsr::i<DataVector, DIM(data), FRAME(data)>& half_phi_two_normals, \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& spacetime_metric,    \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& pi,                  \
      const tnsr::iaa<DataVector, DIM(data), FRAME(data)>& phi,                \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& coords,               \
      const double amp_coef_L1, const double amp_coef_L2,                      \
      const double amp_coef_S, const int exp_L1, const int exp_L2,             \
      const int exp_S, const double sigma_r);

GENERATE_INSTANTIATIONS(INSTANTIATE_DV_FUNC, (1, 2, 3), (DataVector),
                        (Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef DTYPE_SCAL
#undef INSTANTIATE_DV_FUNC
#undef INSTANTIATE_SCALAR_FUNC
}  // namespace gh::gauges
