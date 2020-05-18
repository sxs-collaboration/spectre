// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedWaveHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace GeneralizedHarmonic::gauges {
namespace DampedHarmonicGauge_detail {
// Roll-on function for the damped harmonic gauge.
//
// For times after \f$t_0\f$, compute the roll-on function
// \f$ R(t) = 1 - \exp(-((t - t_0)/\sigma_t)^4]) \f$,
// and return \f$ R(t) = 0\f$ at times before.
double roll_on_function(const double time, const double t_start,
                        const double sigma_t) noexcept {
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
                                      const double sigma_t) noexcept {
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
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, const double time,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double amp_coef_L1, const double amp_coef_L2, const double amp_coef_S,
    const int exp_L1, const int exp_L2, const int exp_S,
    const double t_start_h_init, const double sigma_t_h_init,
    const double t_start_L1, const double sigma_t_L1, const double t_start_L2,
    const double sigma_t_L2, const double t_start_S, const double sigma_t_S,
    const double sigma_r) noexcept {
  destructive_resize_components(gauge_h, get(lapse).size());
  destructive_resize_components(d4_gauge_h, get(lapse).size());

  if constexpr (UseRollon) {
    ASSERT(gauge_h_init != nullptr,
           "Cannot call damped_harmonic_impl with UseRollon enabled and "
           "gauge_h_init being nullptr");
    ASSERT(dgauge_h_init != nullptr,
           "Cannot call damped_harmonic_impl with UseRollon enabled and "
           "dgauge_h_init being nullptr");
  }

  // Use a TempBuffer to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  TempBuffer<tmpl::list<
      ::Tags::TempA<0, SpatialDim, Frame>, ::Tags::Tempii<1, SpatialDim, Frame>,
      ::Tags::TempAA<2, SpatialDim, Frame>,
      ::Tags::Tempii<3, SpatialDim, Frame>,
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
      ::Tags::Tempa<24, SpatialDim, Frame>, ::Tags::TempScalar<25>,
      ::Tags::Tempi<26, SpatialDim, Frame>,
      ::Tags::Tempa<27, SpatialDim, Frame>,
      ::Tags::TempI<28, SpatialDim, Frame>,
      ::Tags::TempiJ<29, SpatialDim, Frame>,
      ::Tags::TempaB<30, SpatialDim, Frame>,
      ::Tags::Tempab<31, SpatialDim, Frame>, ::Tags::TempScalar<32>,
      ::Tags::Tempa<33, SpatialDim, Frame>,
      ::Tags::Tempabb<34, SpatialDim, Frame>,
      ::Tags::Tempab<35, SpatialDim, Frame>,
      ::Tags::Tempab<36, SpatialDim, Frame>,
      ::Tags::Tempab<37, SpatialDim, Frame>>>
      buffer(get_size(get(lapse)));
  auto& spacetime_unit_normal =
      get<::Tags::TempA<0, SpatialDim, Frame>>(buffer);
  auto& spatial_metric = get<::Tags::Tempii<1, SpatialDim, Frame>>(buffer);
  auto& inverse_spacetime_metric =
      get<::Tags::TempAA<2, SpatialDim, Frame>>(buffer);
  auto& d0_spatial_metric = get<::Tags::Tempii<3, SpatialDim, Frame>>(buffer);
  auto& d3_spatial_metric = get<::Tags::Tempijj<4, SpatialDim, Frame>>(buffer);
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
  auto& dt_lapse = get<::Tags::TempScalar<25>>(buffer);
  auto& d3_lapse = get<::Tags::Tempi<26, SpatialDim, Frame>>(buffer);
  auto& d4_lapse = get<::Tags::Tempa<27, SpatialDim, Frame>>(buffer);
  auto& d0_shift = get<::Tags::TempI<28, SpatialDim, Frame>>(buffer);
  auto& d3_shift = get<::Tags::TempiJ<29, SpatialDim, Frame>>(buffer);
  auto& d4_shift = get<::Tags::TempaB<30, SpatialDim, Frame>>(buffer);
  auto& d4_normal_one_form = get<::Tags::Tempab<31, SpatialDim, Frame>>(buffer);
  auto& prefac = get<::Tags::TempScalar<32>>(buffer);
  auto& d4_muS_over_lapse = get<::Tags::Tempa<33, SpatialDim, Frame>>(buffer);
  auto& d4_psi = get<::Tags::Tempabb<34, SpatialDim, Frame>>(buffer);
  auto& dT1 = get<::Tags::Tempab<35, SpatialDim, Frame>>(buffer);
  auto& dT2 = get<::Tags::Tempab<36, SpatialDim, Frame>>(buffer);
  auto& dT3 = get<::Tags::Tempab<37, SpatialDim, Frame>>(buffer);

  // 3+1 quantities
  spacetime_unit_normal =
      gr::spacetime_normal_vector<SpatialDim, Frame, DataVector>(lapse, shift);
  spatial_metric =
      gr::spatial_metric<SpatialDim, Frame, DataVector>(spacetime_metric);
  inverse_spacetime_metric =
      gr::inverse_spacetime_metric<SpatialDim, Frame, DataVector>(
          lapse, shift, inverse_spatial_metric);
  time_deriv_of_spatial_metric<SpatialDim, Frame, DataVector>(
      make_not_null(&d0_spatial_metric), lapse, shift, phi, pi);
  deriv_spatial_metric<SpatialDim, Frame, DataVector>(
      make_not_null(&d3_spatial_metric), phi);

  // commonly used terms
  constexpr auto exp_fac_1 = 0.5;
  constexpr auto exp_fac_2 = 0.;
  get(one_over_lapse) = 1. / get(lapse);
  DampedHarmonicGauge_detail::log_factor_metric_lapse<DataVector>(
      make_not_null(&log_fac_1), lapse, sqrt_det_spatial_metric, exp_fac_1);
  DampedHarmonicGauge_detail::log_factor_metric_lapse<DataVector>(
      make_not_null(&log_fac_2), lapse, sqrt_det_spatial_metric, exp_fac_2);

  // Tempering functions
  const auto roll_on_h_init = DampedHarmonicGauge_detail::roll_on_function(
      time, t_start_h_init, sigma_t_h_init);
  const auto roll_on_L1 = DampedHarmonicGauge_detail::roll_on_function(
      time, t_start_L1, sigma_t_L1);
  const auto roll_on_L2 = DampedHarmonicGauge_detail::roll_on_function(
      time, t_start_L2, sigma_t_L2);
  const auto roll_on_S =
      DampedHarmonicGauge_detail::roll_on_function(time, t_start_S, sigma_t_S);
  DampedHarmonicGauge_detail::spatial_weight_function<SpatialDim, Frame,
                                                      DataVector>(
      make_not_null(&weight), coords, sigma_r);

  // coeffs that enter gauge source function
  get(mu_L1) =
      amp_coef_L1 * roll_on_L1 * get(weight) * pow(get(log_fac_1), exp_L1);
  get(mu_S) = amp_coef_S * roll_on_S * get(weight) * pow(get(log_fac_1), exp_S);
  get(mu_L2) =
      amp_coef_L2 * roll_on_L2 * get(weight) * pow(get(log_fac_2), exp_L2);
  get(mu_S_over_lapse) = get(mu_S) * get(one_over_lapse);

  // Calc \f$ \mu_1 = \mu_{L1} log(rootg/N) = R W log(rootg/N)^5\f$
  get(mu1) = get(mu_L1) * get(log_fac_1);

  // Calc \f$ \mu_2 = \mu_{L2} log(1/N) = R W log(1/N)^5\f$
  get(mu2) = get(mu_L2) * get(log_fac_2);

  get(prefac) = get(mu_L1) * get(log_fac_1) + get(mu_L2) * get(log_fac_2);

  // Calculate H_a
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    if constexpr (UseRollon) {
      gauge_h->get(a) = (1. - roll_on_h_init) * gauge_h_init->get(a) +
                        get(prefac) * spacetime_unit_normal_one_form.get(a);
    } else {
      gauge_h->get(a) = get(prefac) * spacetime_unit_normal_one_form.get(a);
    }
    for (size_t i = 0; i < SpatialDim; ++i) {
      gauge_h->get(a) -=
          get(mu_S_over_lapse) * spacetime_metric.get(a, i + 1) * shift.get(i);
    }
  }

  const auto d0_roll_on_h_init =
      DampedHarmonicGauge_detail::time_deriv_of_roll_on_function(
          time, t_start_h_init, sigma_t_h_init);
  const auto d0_roll_on_L1 =
      DampedHarmonicGauge_detail::time_deriv_of_roll_on_function(
          time, t_start_L1, sigma_t_L1);
  const auto d0_roll_on_S =
      DampedHarmonicGauge_detail::time_deriv_of_roll_on_function(
          time, t_start_S, sigma_t_S);
  const auto d0_roll_on_L2 =
      DampedHarmonicGauge_detail::time_deriv_of_roll_on_function(
          time, t_start_L2, sigma_t_L2);

  // Calc \f$ \partial_a [R W] \f$
  DampedHarmonicGauge_detail::spacetime_deriv_of_spatial_weight_function<
      SpatialDim, Frame, DataVector>(make_not_null(&d4_weight), coords, sigma_r,
                                     weight);
  d4_RW_L1 = d4_weight;
  d4_RW_S = d4_weight;
  d4_RW_L2 = d4_weight;
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_RW_L1.get(a) *= roll_on_L1;
    d4_RW_S.get(a) *= roll_on_S;
    d4_RW_L2.get(a) *= roll_on_L2;
  }
  get<0>(d4_RW_L1) += get(weight) * d0_roll_on_L1;
  get<0>(d4_RW_S) += get(weight) * d0_roll_on_S;
  get<0>(d4_RW_L2) += get(weight) * d0_roll_on_L2;

  // \partial_a \mu_{S} = \partial_a(A_S R_S W
  //                               \log(\sqrt{g}/N)^{c_{S}})
  // \partial_a \mu_1 = \partial_a(A_L1 R_L1 W
  //                               \log(\sqrt{g}/N)^{1+c_{L1}})
  // \partial_a \mu_2 = \partial_a(A_L2 R_L2 W
  //                               \log(1/N)^{1+c_{L2}})
  DampedHarmonicGauge_detail::spacetime_deriv_of_power_log_factor_metric_lapse<
      SpatialDim, Frame, DataVector>(
      make_not_null(&d4_log_fac_mu1), lapse, shift, spacetime_unit_normal,
      inverse_spatial_metric, sqrt_det_spatial_metric, d0_spatial_metric, pi,
      phi, exp_fac_1, exp_L1 + 1);
  DampedHarmonicGauge_detail::spacetime_deriv_of_power_log_factor_metric_lapse<
      SpatialDim, Frame, DataVector>(
      make_not_null(&d4_log_fac_muS), lapse, shift, spacetime_unit_normal,
      inverse_spatial_metric, sqrt_det_spatial_metric, d0_spatial_metric, pi,
      phi, exp_fac_1, exp_S);
  DampedHarmonicGauge_detail::spacetime_deriv_of_power_log_factor_metric_lapse<
      SpatialDim, Frame, DataVector>(
      make_not_null(&d4_log_fac_mu2), lapse, shift, spacetime_unit_normal,
      inverse_spatial_metric, sqrt_det_spatial_metric, d0_spatial_metric, pi,
      phi, exp_fac_2, exp_L2 + 1);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    // \f$ \partial_a \mu_1 \f$
    d4_mu1.get(a) =
        amp_coef_L1 * pow(get(log_fac_1), exp_L1 + 1) * d4_RW_L1.get(a) +
        amp_coef_L1 * roll_on_L1 * get(weight) * d4_log_fac_mu1.get(a);
    // \f$ \partial_a \mu_{S} \f$
    d4_mu_S.get(a) =
        amp_coef_S * d4_RW_S.get(a) * pow(get(log_fac_1), exp_S) +
        amp_coef_S * roll_on_S * get(weight) * d4_log_fac_muS.get(a);
    // \f$ \partial_a \mu_2 \f$
    d4_mu2.get(a) =
        amp_coef_L2 * pow(get(log_fac_2), exp_L2 + 1) * d4_RW_L2.get(a) +
        amp_coef_L2 * roll_on_L2 * get(weight) * d4_log_fac_mu2.get(a);
  }

  // Calc \f$ \partial_a N = {\partial_0 N, \partial_i N} \f$
  time_deriv_of_lapse<SpatialDim, Frame, DataVector>(
      make_not_null(&dt_lapse), lapse, shift, spacetime_unit_normal, phi, pi);
  spatial_deriv_of_lapse<SpatialDim, Frame, DataVector>(
      make_not_null(&d3_lapse), lapse, spacetime_unit_normal, phi);
  get<0>(d4_lapse) = get(dt_lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_lapse.get(1 + i) = d3_lapse.get(i);
  }

  // Calc \f$ \partial_a N^i = {\partial_0 N^i, \partial_j N^i} \f$
  d0_shift = time_deriv_of_shift<SpatialDim, Frame, DataVector>(
      lapse, shift, inverse_spatial_metric, spacetime_unit_normal, phi, pi);
  d3_shift = spatial_deriv_of_shift<SpatialDim, Frame, DataVector>(
      lapse, inverse_spacetime_metric, spacetime_unit_normal, phi);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_shift.get(0, 1 + i) = d0_shift.get(i);
    for (size_t j = 0; j < SpatialDim; ++j) {
      d4_shift.get(1 + j, 1 + i) = d3_shift.get(j, i);
    }
  }

  // Calc \f$ \partial_a n_b = {-\partial_a N, 0, 0, 0} \f$
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_normal_one_form.get(a, 0) = -d4_lapse.get(a);
    for (size_t b = 1; b < SpatialDim + 1; ++b) {
      d4_normal_one_form.get(a, b) = 0.;
    }
  }

  // \f[ \partial_a (\mu_S/N) = (1/N) \partial_a \mu_{S}
  //         - (\mu_{S}/N^2) \partial_a N
  // \f]
  get(prefac) = -get(mu_S) * square(get(one_over_lapse));
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_muS_over_lapse.get(a) =
        get(one_over_lapse) * d4_mu_S.get(a) + get(prefac) * d4_lapse.get(a);
  }

  // We need \f$ \partial_a g_{bi} = \partial_a \psi_{bi} \f$. Here we
  // use `derivatives_of_spacetime_metric` to get \f$ \partial_a g_{bc}\f$
  // instead, and use only the derivatives of \f$ g_{bi}\f$.
  d4_psi = gr::derivatives_of_spacetime_metric<SpatialDim, Frame, DataVector>(
      lapse, dt_lapse, d3_lapse, shift, d0_shift, d3_shift, spatial_metric,
      d0_spatial_metric, d3_spatial_metric);

  // Calc \f$ \partial_a T1 \f$
  if constexpr (UseRollon) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      dT1.get(0, b) = -gauge_h_init->get(b) * d0_roll_on_h_init;
      for (size_t a = 0; a < SpatialDim + 1; ++a) {
        if (a != 0) {
          dT1.get(a, b) = (1. - roll_on_h_init) * dgauge_h_init->get(a, b);
        } else {
          dT1.get(a, b) += (1. - roll_on_h_init) * dgauge_h_init->get(a, b);
        }
      }
    }
  }

  // Calc \f$ \partial_a T2 \f$
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      dT2.get(a, b) = (get(mu1) + get(mu2)) * d4_normal_one_form.get(a, b) +
                      (d4_mu1.get(a) + d4_mu2.get(a)) *
                          spacetime_unit_normal_one_form.get(b);
    }
  }

  // Calc \f$ \partial_a T3 \f$ (note minus sign)
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      dT3.get(a, b) = -d4_muS_over_lapse.get(a) * get<0>(shift) *
                      spacetime_metric.get(b, 1);
      for (size_t i = 0; i < SpatialDim; ++i) {
        if (i != 0) {
          dT3.get(a, b) -= d4_muS_over_lapse.get(a) * shift.get(i) *
                           spacetime_metric.get(b, i + 1);
        }
        dT3.get(a, b) -=
            get(mu_S_over_lapse) * shift.get(i) * d4_psi.get(a, b, i + 1);
        dT3.get(a, b) -= get(mu_S_over_lapse) * spacetime_metric.get(b, i + 1) *
                         d4_shift.get(a, i + 1);
      }
    }
  }

  // Calc \f$ \partial_a H_b = dT1_{ab} + dT2_{ab} + dT3_{ab} \f$
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      if constexpr (UseRollon) {
        d4_gauge_h->get(a, b) = dT1.get(a, b) + dT2.get(a, b) + dT3.get(a, b);
      } else {
        d4_gauge_h->get(a, b) = dT2.get(a, b) + dT3.get(a, b);
      }
    }
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
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, const double time,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double amp_coef_L1, const double amp_coef_L2, const double amp_coef_S,
    const int exp_L1, const int exp_L2, const int exp_S,
    const double t_start_h_init, const double sigma_t_h_init,
    const double t_start_L1, const double sigma_t_L1, const double t_start_L2,
    const double sigma_t_L2, const double t_start_S, const double sigma_t_S,
    const double sigma_r) noexcept {
  damped_harmonic_impl<true>(
      gauge_h, d4_gauge_h, &gauge_h_init, &dgauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, time, coords,
      amp_coef_L1, amp_coef_L2, amp_coef_S, exp_L1, exp_L2, exp_S,
      t_start_h_init, sigma_t_h_init, t_start_L1, sigma_t_L1, t_start_L2,
      sigma_t_L2, t_start_S, sigma_t_S, sigma_r);
}

template <size_t SpatialDim, typename Frame>
void DampedHarmonicRollonCompute<SpatialDim, Frame>::function(
    const gsl::not_null<return_type*> h_and_d4_h,
    const db::item_type<Tags::InitialGaugeH<SpatialDim, Frame>>& gauge_h_init,
    const db::item_type<Tags::SpacetimeDerivInitialGaugeH<SpatialDim, Frame>>&
        dgauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const tnsr::a<DataVector, SpatialDim, Frame>&
        spacetime_unit_normal_one_form,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, const double time,
    const double t_start, const double sigma_t,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double sigma_r) noexcept {
  if (UNLIKELY(h_and_d4_h->number_of_grid_points() != get(lapse).size())) {
    h_and_d4_h->initialize(get(lapse).size());
  }
  // exp_{L1, L2, S}
  // This should be read from the input file in the future.
  constexpr int exponent = 4;
  damped_harmonic_rollon(
      make_not_null(&get<Tags::GaugeH<SpatialDim, Frame>>(*h_and_d4_h)),
      make_not_null(
          &get<Tags::SpacetimeDerivGaugeH<SpatialDim, Frame>>(*h_and_d4_h)),
      gauge_h_init, dgauge_h_init, lapse, shift, spacetime_unit_normal_one_form,
      sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric, pi,
      phi, time, coords, 1., 1.,
      1.,                            // amp_coef_{L1, L2, S}
      exponent, exponent, exponent,  // exp_{L1, L2, S}
      t_start, sigma_t,              // _h_init
      t_start, sigma_t,              // _L1
      t_start, sigma_t,              // _L2
      t_start, sigma_t,              // _S
      sigma_r);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define DTYPE_SCAL(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_DV_FUNC(_, data)                                           \
  template void damped_harmonic_rollon(                                        \
      const gsl::not_null<tnsr::a<DataVector, DIM(data), FRAME(data)>*>        \
          gauge_h,                                                             \
      const gsl::not_null<tnsr::ab<DataVector, DIM(data), FRAME(data)>*>       \
          d4_gauge_h,                                                          \
      const tnsr::a<DataVector, DIM(data), FRAME(data)>& gauge_h_init,         \
      const tnsr::ab<DataVector, DIM(data), FRAME(data)>& dgauge_h_init,       \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& shift,                \
      const tnsr::a<DataVector, DIM(data), FRAME(data)>&                       \
          spacetime_unit_normal_one_form,                                      \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                       \
      const tnsr::II<DataVector, DIM(data), FRAME(data)>&                      \
          inverse_spatial_metric,                                              \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& spacetime_metric,    \
      const tnsr::aa<DataVector, DIM(data), FRAME(data)>& pi,                  \
      const tnsr::iaa<DataVector, DIM(data), FRAME(data)>& phi,                \
      const double time,                                                       \
      const tnsr::I<DataVector, DIM(data), FRAME(data)>& coords,               \
      const double amp_coef_L1, const double amp_coef_L2,                      \
      const double amp_coef_S, const int exp_L1, const int exp_L2,             \
      const int exp_S, const double t_start_h_init,                            \
      const double sigma_t_h_init, const double t_start_L1,                    \
      const double sigma_t_L1, const double t_start_L2,                        \
      const double sigma_t_L2, const double t_start_S, const double sigma_t_S, \
      const double sigma_r) noexcept;                                          \
  template class DampedHarmonicRollonCompute<DIM(data), FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE_DV_FUNC, (1, 2, 3), (DataVector),
                        (Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef DTYPE_SCAL
#undef INSTANTIATE_DV_FUNC
#undef INSTANTIATE_SCALAR_FUNC
/// \endcond
}  // namespace GeneralizedHarmonic::gauges
