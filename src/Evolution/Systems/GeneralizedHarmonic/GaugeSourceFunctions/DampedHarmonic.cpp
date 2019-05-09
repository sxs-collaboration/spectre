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
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
namespace DampedHarmonicGauge_detail {
// Spatial weight function used in the damped harmonic gauge source
// function.
//
// The spatial weight function is: \f[ W(x^i) = \exp(- (r / \sigma_r)^2) \f]
// This function can be written with an extra factor inside the exponent in
// literature, e.g. \cite Deppe2018uye. We absorb that in \f$\sigma_r\f$.
template <size_t SpatialDim, typename Frame, typename DataType>
void weight_function(const gsl::not_null<Scalar<DataType>*> weight,
                     const tnsr::I<DataType, SpatialDim, Frame>& coords,
                     const double sigma_r) noexcept {
  if (UNLIKELY(get_size(get(*weight)) != get_size(get<0>(coords)))) {
    *weight = Scalar<DataType>(get_size(get<0>(coords)));
  }
  const auto& r_squared = dot_product(coords, coords);
  get(*weight) = exp(-get(r_squared) / pow<2>(sigma_r));
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> weight_function(
    const tnsr::I<DataType, SpatialDim, Frame>& coords,
    const double sigma_r) noexcept {
  Scalar<DataType> weight{};
  GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function(
      make_not_null(&weight), coords, sigma_r);
  return weight;
}

// Spacetime derivatives of the spatial weight function that enters the
// damped harmonic gauge source function.
//
// Compute the derivatives :
// \partial_a W(x^i)= \partial_a \exp(- (r/\sigma_r)^2)
//                  = (-2 * x^i / \sigma_r^2) * exp(-(r/\sigma_r)^2)
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_weight_function(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_weight,
    const tnsr::I<DataType, SpatialDim, Frame>& coords,
    const double sigma_r) noexcept {
  if (UNLIKELY(get_size(get<0>(*d4_weight)) != get_size(get<0>(coords)))) {
    *d4_weight = tnsr::a<DataType, SpatialDim, Frame>(get_size(get<0>(coords)));
  }
  const DataType pre_factor =
      get(weight_function(coords, sigma_r)) * (-2. / pow<2>(sigma_r));
  // time derivative of weight function is zero
  get<0>(*d4_weight) = 0.;
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_weight->get(1 + i) = pre_factor * coords.get(i);
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_weight_function(
    const tnsr::I<DataType, SpatialDim, Frame>& coords,
    const double sigma_r) noexcept {
  tnsr::a<DataType, SpatialDim, Frame> d4_weight{};
  GeneralizedHarmonic::DampedHarmonicGauge_detail::
      spacetime_deriv_of_weight_function(make_not_null(&d4_weight), coords,
                                         sigma_r);
  return d4_weight;
}

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

// The log factor that appears in damped harmonic gauge source function.
//
// Calculates:  \f$ logF = \mathrm{log}(g^p/N) \f$.
template <typename DataType>
void log_factor_metric_lapse(const gsl::not_null<Scalar<DataType>*> logfac,
                             const Scalar<DataType>& lapse,
                             const Scalar<DataType>& sqrt_det_spatial_metric,
                             const double exponent) noexcept {
  if (UNLIKELY(get_size(get(*logfac)) != get_size(get(lapse)))) {
    *logfac = Scalar<DataType>(get_size(get(lapse)));
  }
  // branching below is to avoid using pow for performance reasons
  if (exponent == 0.) {
    get(*logfac) = -log(get(lapse));
  } else if (exponent == 0.5) {
    get(*logfac) = log(get(sqrt_det_spatial_metric) / get(lapse));
  } else {
    get(*logfac) =
        2. * exponent * log(get(sqrt_det_spatial_metric)) - log(get(lapse));
  }
}

template <typename DataType>
Scalar<DataType> log_factor_metric_lapse(
    const Scalar<DataType>& lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const double exponent) noexcept {
  Scalar<DataType> logfac{};
  GeneralizedHarmonic::DampedHarmonicGauge_detail::log_factor_metric_lapse(
      make_not_null(&logfac), lapse, sqrt_det_spatial_metric, exponent);
  return logfac;
}

// Spacetime derivatives of the log factor that appears in the
// damped harmonic gauge source function.
//
// Computes the spacetime derivatives:
//  \partial_a logF = (p/g)\partial_a g - (1/N)\partial_a N
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_log_factor_metric_lapse(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_logfac,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const double exponent) noexcept {
  if (UNLIKELY(get_size(get<0>(*d4_logfac)) != get_size(get(lapse)))) {
    *d4_logfac = tnsr::a<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  // Use a TempBuffer to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  TempBuffer<tmpl::list<::Tags::Tempa<0, SpatialDim, Frame, DataType>,
                        ::Tags::Tempi<1, SpatialDim, Frame, DataType>,
                        ::Tags::Tempa<2, SpatialDim, Frame, DataType>,
                        ::Tags::TempScalar<3, DataType>,
                        ::Tags::TempScalar<4, DataType>>>
      buffer(get_size(get(lapse)));
  auto& d_g = get<::Tags::Tempa<0, SpatialDim, Frame, DataType>>(buffer);
  auto& d3_lapse = get<::Tags::Tempi<1, SpatialDim, Frame, DataType>>(buffer);
  auto& d_lapse = get<::Tags::Tempa<2, SpatialDim, Frame, DataType>>(buffer);
  auto& dt_lapse = get<::Tags::TempScalar<3, DataType>>(buffer);
  auto& one_over_lapse = get<::Tags::TempScalar<4, DataType>>(buffer);

  // Get \f$ \partial_a g\f$
  spacetime_deriv_of_det_spatial_metric<SpatialDim, Frame, DataType>(
      make_not_null(&d_g), sqrt_det_spatial_metric, inverse_spatial_metric,
      dt_spatial_metric, phi);
  // Get \f$ \partial_a N\f$
  time_deriv_of_lapse<SpatialDim, Frame, DataType>(
      make_not_null(&dt_lapse), lapse, shift, spacetime_unit_normal, phi, pi);
  spatial_deriv_of_lapse<SpatialDim, Frame, DataType>(
      make_not_null(&d3_lapse), lapse, spacetime_unit_normal, phi);
  get<0>(d_lapse) = get(dt_lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d_lapse.get(1 + i) = d3_lapse.get(i);
  }
  // Compute
  get(one_over_lapse) = 1. / get(lapse);
  if (exponent == 0.) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_logfac->get(a) = -get(one_over_lapse) * d_lapse.get(a);
    }
  } else {
    const auto p_over_g = exponent / square(get(sqrt_det_spatial_metric));
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_logfac->get(a) =
          p_over_g * d_g.get(a) - get(one_over_lapse) * d_lapse.get(a);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_log_factor_metric_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const double exponent) noexcept {
  tnsr::a<DataType, SpatialDim, Frame> d4_logfac{};
  GeneralizedHarmonic::DampedHarmonicGauge_detail::
      spacetime_deriv_of_log_factor_metric_lapse(
          make_not_null(&d4_logfac), lapse, shift, spacetime_unit_normal,
          inverse_spatial_metric, sqrt_det_spatial_metric, dt_spatial_metric,
          pi, phi, exponent);
  return d4_logfac;
}

// Spacetime derivatives of the log factor (that appears in
// damped harmonic gauge source function), raised to an exponent.
//
// Computes the spacetime derivatives:
//  \partial_a (logF)^q = q (logF)^{q-1} \partial_a (logF)
template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_power_log_factor_metric_lapse(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_powlogfac,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi, const double g_exponent,
    const int exponent) noexcept {
  if (UNLIKELY(get_size(get<0>(*d4_powlogfac)) != get_size(get(lapse)))) {
    *d4_powlogfac = tnsr::a<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  // Use a TempBuffer to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  TempBuffer<tmpl::list<::Tags::Tempa<0, SpatialDim, Frame, DataType>,
                        ::Tags::TempScalar<1, DataType>,
                        ::Tags::TempScalar<2, DataType>>>
      buffer(get_size(get(lapse)));
  auto& dlogfac = get<::Tags::Tempa<0, SpatialDim, Frame, DataType>>(buffer);
  auto& logfac = get<::Tags::TempScalar<1, DataType>>(buffer);
  auto& prefac = get<::Tags::TempScalar<2, DataType>>(buffer);

  // Compute derivative
  spacetime_deriv_of_log_factor_metric_lapse<SpatialDim, Frame, DataType>(
      make_not_null(&dlogfac), lapse, shift, spacetime_unit_normal,
      inverse_spatial_metric, sqrt_det_spatial_metric, dt_spatial_metric, pi,
      phi, g_exponent);
  // Apply pre-factor
  if (UNLIKELY(exponent == 0)) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_powlogfac->get(a) = 0.;
    }
  } else if (UNLIKELY(exponent == 1)) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_powlogfac->get(a) = dlogfac.get(a);
    }
  } else {
    log_factor_metric_lapse<DataType>(make_not_null(&logfac), lapse,
                                      sqrt_det_spatial_metric, g_exponent);
    get(prefac) =
        static_cast<double>(exponent) * pow(get(logfac), exponent - 1);
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_powlogfac->get(a) = get(prefac) * dlogfac.get(a);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame>
spacetime_deriv_of_power_log_factor_metric_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi, const double g_exponent,
    const int exponent) noexcept {
  tnsr::a<DataType, SpatialDim, Frame> d4_powlogfac{};
  GeneralizedHarmonic::DampedHarmonicGauge_detail::
      spacetime_deriv_of_power_log_factor_metric_lapse(
          make_not_null(&d4_powlogfac), lapse, shift, spacetime_unit_normal,
          inverse_spatial_metric, sqrt_det_spatial_metric, dt_spatial_metric,
          pi, phi, g_exponent, exponent);
  return d4_powlogfac;
}
}  // namespace DampedHarmonicGauge_detail

// Assemble the gauge source function.
//
// Recall that its covariant form is:
// H_a := [1 - R_{H_\mathrm{init}}(t)] H_a^\mathrm{init} +
//  [\mu_{L1} log(\sqrt{g}/N) + \mu_{L2} log(1/N)] t_a - \mu_S g_{ai} N^i / N
//
// where \f$N, N^k\f$ are the lapse and shift, and \f$n_k\f$ is the unit
// normal one-form to the spatial slice, as above. The pre-factors:
//
//  \mu_{L1} = A_{L1} R_{L1}(t) W(x^i) log(\sqrt{g}/N)^{e_{L1}},
//  \mu_{L2} = A_{L2} R_{L2}(t) W(x^i) log(1/N)^{e_{L2}},
//  \mu_{S} = A_{S} R_{S}(t) W(x^i) log(\sqrt{g}/N)^{e_{S}},
//
// contain the temporal and spatial roll-on functions.
template <size_t SpatialDim, typename Frame>
void damped_harmonic_h(
    const gsl::not_null<
        typename db::item_type<Tags::GaugeH<SpatialDim, Frame>>*>
        gauge_h,
    const typename db::item_type<Tags::InitialGaugeH<SpatialDim, Frame>>&
        gauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const double time, const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double amp_coef_L1, const double amp_coef_L2, const double amp_coef_S,
    const int exp_L1, const int exp_L2, const int exp_S,
    const double t_start_h_init, const double sigma_t_h_init,
    const double t_start_L1, const double sigma_t_L1, const double t_start_L2,
    const double sigma_t_L2, const double t_start_S, const double sigma_t_S,
    const double sigma_r) noexcept {
  if (UNLIKELY(get_size(get<0>(*gauge_h)) != get_size(get(lapse)))) {
    *gauge_h = typename db::item_type<Tags::GaugeH<SpatialDim, Frame>>(
        get_size(get(lapse)));
  }
  // Use a TempBuffer to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  TempBuffer<tmpl::list<::Tags::Tempa<0, SpatialDim, Frame>,
                        ::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
                        ::Tags::TempScalar<3>, ::Tags::TempScalar<4>,
                        ::Tags::TempScalar<5>, ::Tags::TempScalar<6>,
                        ::Tags::TempScalar<7>, ::Tags::TempScalar<8>>>
      buffer(get_size(get(lapse)));
  auto& spacetime_unit_normal_one_form =
      get<::Tags::Tempa<0, SpatialDim, Frame>>(buffer);
  auto& log_fac_1 = get<::Tags::TempScalar<1>>(buffer);
  auto& log_fac_2 = get<::Tags::TempScalar<2>>(buffer);
  auto& weight = get<::Tags::TempScalar<3>>(buffer);
  auto& mu_L1 = get<::Tags::TempScalar<4>>(buffer);
  auto& mu_S = get<::Tags::TempScalar<5>>(buffer);
  auto& mu_L2 = get<::Tags::TempScalar<6>>(buffer);
  auto& h_prefac1 = get<::Tags::TempScalar<7>>(buffer);
  auto& h_prefac2 = get<::Tags::TempScalar<8>>(buffer);

  spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame, DataVector>(lapse);

  constexpr double exp_fac_1 = 0.5;
  constexpr double exp_fac_2 = 0.;
  DampedHarmonicGauge_detail::log_factor_metric_lapse<DataVector>(
      make_not_null(&log_fac_1), lapse, sqrt_det_spatial_metric, exp_fac_1);
  DampedHarmonicGauge_detail::log_factor_metric_lapse<DataVector>(
      make_not_null(&log_fac_2), lapse, sqrt_det_spatial_metric, exp_fac_2);

  const double roll_on_L1 = DampedHarmonicGauge_detail::roll_on_function(
      time, t_start_L1, sigma_t_L1);
  const double roll_on_L2 = DampedHarmonicGauge_detail::roll_on_function(
      time, t_start_L2, sigma_t_L2);
  const double roll_on_S =
      DampedHarmonicGauge_detail::roll_on_function(time, t_start_S, sigma_t_S);
  DampedHarmonicGauge_detail::weight_function<SpatialDim, Frame, DataVector>(
      make_not_null(&weight), coords, sigma_r);

  get(mu_L1) =
      amp_coef_L1 * roll_on_L1 * get(weight) * pow(get(log_fac_1), exp_L1);
  get(mu_S) = amp_coef_S * roll_on_S * get(weight) * pow(get(log_fac_1), exp_S);
  get(mu_L2) =
      amp_coef_L2 * roll_on_L2 * get(weight) * pow(get(log_fac_2), exp_L2);
  get(h_prefac1) = get(mu_L1) * get(log_fac_1) + get(mu_L2) * get(log_fac_2);
  get(h_prefac2) = -get(mu_S) / get(lapse);

  const double roll_on_h_init = DampedHarmonicGauge_detail::roll_on_function(
      time, t_start_h_init, sigma_t_h_init);

  // Calculate H_a
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    gauge_h->get(a) = (1. - roll_on_h_init) * gauge_h_init.get(a) +
                      get(h_prefac1) * spacetime_unit_normal_one_form.get(a);
    for (size_t i = 0; i < SpatialDim; ++i) {
      gauge_h->get(a) +=
          get(h_prefac2) * spacetime_metric.get(a, i + 1) * shift.get(i);
    }
  }
}

template <size_t SpatialDim, typename Frame>
typename db::item_type<Tags::GaugeH<SpatialDim, Frame>>
DampedHarmonicHCompute<SpatialDim, Frame>::function(
    const typename db::item_type<Tags::InitialGaugeH<SpatialDim, Frame>>&
        gauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const Time& time, const double& t_start, const double& sigma_t,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double& sigma_r) noexcept {
  typename db::item_type<Tags::GaugeH<SpatialDim, Frame>> gauge_h{
      get_size(get(lapse))};
  GeneralizedHarmonic::damped_harmonic_h<SpatialDim, Frame>(
      make_not_null(&gauge_h), gauge_h_init, lapse, shift,
      sqrt_det_spatial_metric, spacetime_metric, time.value(), coords, 1., 1.,
      1.,                // amp_coef_{L1, L2, S}
      4, 4, 4,           // exp_{L1, L2, S}
      t_start, sigma_t,  // _h_init
      t_start, sigma_t,  // _L1
      t_start, sigma_t,  // _L2
      t_start, sigma_t,  // _S
      sigma_r);
  return gauge_h;
}

// Spacetime derivatives of the damped harmonic gauge source function.
//
// The following functions and struct compute spacetime derivatives, i.e.
// \f$\partial_a H_b\f$, of the damped hamornic source function H. From above:
//
// \partial_a H_b = \partial_a T_1 + \partial_a T_2 + \partial_a T_3
// H_a = T_1 + T_2 + T_3,
//
// where:
//
// T_1 = [1 - R_{H_\mathrm{init}}(t)] H_a^\mathrm{init},
// T_2 = [\mu_{L1} log(\sqrt{g}/N) + \mu_{L2} log(1/N)] t_a,
// T_3 = - \mu_S g_{ai} N^i / N.
//
// See the header file for \f$\partial_a T1,2,3 \f$.
template <size_t SpatialDim, typename Frame>
void spacetime_deriv_damped_harmonic_h(
    const gsl::not_null<
        typename db::item_type<Tags::SpacetimeDerivGaugeH<SpatialDim, Frame>>*>
        d4_gauge_h,
    const typename db::item_type<Tags::InitialGaugeH<SpatialDim, Frame>>&
        gauge_h_init,
    const typename db::item_type<
        Tags::SpacetimeDerivInitialGaugeH<SpatialDim, Frame>>& dgauge_h_init,
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
  if (UNLIKELY(get_size(get<0, 0>(*d4_gauge_h)) != get_size(get(lapse)))) {
    *d4_gauge_h =
        typename db::item_type<Tags::SpacetimeDerivGaugeH<SpatialDim, Frame>>(
            get(lapse));
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
  DampedHarmonicGauge_detail::weight_function<SpatialDim, Frame, DataVector>(
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
  DampedHarmonicGauge_detail::spacetime_deriv_of_weight_function<
      SpatialDim, Frame, DataVector>(make_not_null(&d4_weight), coords,
                                     sigma_r);
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
  for (size_t b = 0; b < SpatialDim + 1; ++b) {
    dT1.get(0, b) = -gauge_h_init.get(b) * d0_roll_on_h_init;
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      if (a != 0) {
        dT1.get(a, b) = (1. - roll_on_h_init) * dgauge_h_init.get(a, b);
      } else {
        dT1.get(a, b) += (1. - roll_on_h_init) * dgauge_h_init.get(a, b);
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
      d4_gauge_h->get(a, b) = dT1.get(a, b) + dT2.get(a, b) + dT3.get(a, b);
    }
  }
}

template <size_t SpatialDim, typename Frame>
typename db::item_type<Tags::SpacetimeDerivGaugeH<SpatialDim, Frame>>
SpacetimeDerivDampedHarmonicHCompute<SpatialDim, Frame>::function(
    const typename db::item_type<Tags::InitialGaugeH<SpatialDim, Frame>>&
        gauge_h_init,
    const typename db::item_type<
        Tags::SpacetimeDerivInitialGaugeH<SpatialDim, Frame>>& dgauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const tnsr::a<DataVector, SpatialDim, Frame>&
        spacetime_unit_normal_one_form,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, const Time& time,
    const double& t_start, const double& sigma_t,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double& sigma_r) noexcept {
  typename db::item_type<Tags::SpacetimeDerivGaugeH<SpatialDim, Frame>>
      d4_gauge_h{get_size(get(lapse))};
  GeneralizedHarmonic::spacetime_deriv_damped_harmonic_h(
      make_not_null(&d4_gauge_h), gauge_h_init, dgauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, time.value(), coords,
      1., 1., 1.,        // amp_coef_{L1, L2, S}
      4, 4, 4,           // exp_{L1, L2, S}
      t_start, sigma_t,  // _h_init
      t_start, sigma_t,  // _L1
      t_start, sigma_t,  // _L2
      t_start, sigma_t,  // _S
      sigma_r);
  return d4_gauge_h;
}
}  // namespace GeneralizedHarmonic

// Explicit Instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define DTYPE_SCAL(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void                                                               \
  GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function(           \
      const gsl::not_null<Scalar<DTYPE(data)>*> weight,                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& coords,             \
      const double sigma_r) noexcept;                                         \
  template Scalar<DTYPE(data)>                                                \
  GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function(           \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& coords,             \
      const double sigma_r) noexcept;                                         \
  template void GeneralizedHarmonic::DampedHarmonicGauge_detail::             \
      spacetime_deriv_of_weight_function(                                     \
          const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>  \
              d4_weight,                                                      \
          const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& coords,         \
          const double sigma_r) noexcept;                                     \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)> GeneralizedHarmonic:: \
      DampedHarmonicGauge_detail::spacetime_deriv_of_weight_function(         \
          const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& coords,         \
          const double sigma_r) noexcept;                                     \
  template void GeneralizedHarmonic::DampedHarmonicGauge_detail::             \
      spacetime_deriv_of_log_factor_metric_lapse(                             \
          const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>  \
              d4_logfac,                                                      \
          const Scalar<DTYPE(data)>& lapse,                                   \
          const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,          \
          const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                 \
              spacetime_unit_normal,                                          \
          const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
              inverse_spatial_metric,                                         \
          const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                 \
          const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
              dt_spatial_metric,                                              \
          const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,            \
          const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,          \
          const double exponent) noexcept;                                    \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)> GeneralizedHarmonic:: \
      DampedHarmonicGauge_detail::spacetime_deriv_of_log_factor_metric_lapse( \
          const Scalar<DTYPE(data)>& lapse,                                   \
          const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,          \
          const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                 \
              spacetime_unit_normal,                                          \
          const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
              inverse_spatial_metric,                                         \
          const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                 \
          const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
              dt_spatial_metric,                                              \
          const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,            \
          const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,          \
          const double exponent) noexcept;                                    \
  template void GeneralizedHarmonic::DampedHarmonicGauge_detail::             \
      spacetime_deriv_of_power_log_factor_metric_lapse(                       \
          const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>  \
              d4_powlogfac,                                                   \
          const Scalar<DTYPE(data)>& lapse,                                   \
          const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,          \
          const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                 \
              spacetime_unit_normal,                                          \
          const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
              inverse_spatial_metric,                                         \
          const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                 \
          const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
              dt_spatial_metric,                                              \
          const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,            \
          const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,          \
          const double g_exponent, const int exponent) noexcept;              \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                       \
  GeneralizedHarmonic::DampedHarmonicGauge_detail::                           \
      spacetime_deriv_of_power_log_factor_metric_lapse(                       \
          const Scalar<DTYPE(data)>& lapse,                                   \
          const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,          \
          const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                 \
              spacetime_unit_normal,                                          \
          const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                \
              inverse_spatial_metric,                                         \
          const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                 \
          const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                \
              dt_spatial_metric,                                              \
          const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,            \
          const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,          \
          const double g_exponent, const int exponent) noexcept;

#define INSTANTIATE_SCALAR_FUNC(_, data)                                    \
  template void                                                             \
  GeneralizedHarmonic::DampedHarmonicGauge_detail::log_factor_metric_lapse( \
      const gsl::not_null<Scalar<DTYPE_SCAL(data)>*> logfac,                \
      const Scalar<DTYPE_SCAL(data)>& lapse,                                \
      const Scalar<DTYPE_SCAL(data)>& sqrt_det_spatial_metric,              \
      const double exponent) noexcept;                                      \
  template Scalar<DTYPE_SCAL(data)>                                         \
  GeneralizedHarmonic::DampedHarmonicGauge_detail::log_factor_metric_lapse( \
      const Scalar<DTYPE_SCAL(data)>& lapse,                                \
      const Scalar<DTYPE_SCAL(data)>& sqrt_det_spatial_metric,              \
      const double exponent) noexcept;

#define INSTANTIATE_COMPUTE_ITEM(_, data)                                    \
  template struct GeneralizedHarmonic::DampedHarmonicHCompute<DIM(data),     \
                                                              FRAME(data)>;  \
  template struct GeneralizedHarmonic::SpacetimeDerivDampedHarmonicHCompute< \
      DIM(data), FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Inertial))

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALAR_FUNC, (double, DataVector))

GENERATE_INSTANTIATIONS(INSTANTIATE_COMPUTE_ITEM, (1, 2, 3), (DataVector),
                        (Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef DTYPE_SCAL
#undef INSTANTIATE
#undef INSTANTIATE_SCALAR_FUNC
#undef INSTANTIATE_COMPUTE_ITEM
/// \endcond
