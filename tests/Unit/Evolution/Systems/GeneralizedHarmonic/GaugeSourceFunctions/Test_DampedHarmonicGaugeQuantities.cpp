// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <random>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
/// \endcond

namespace GeneralizedHarmonic {
namespace DampedHarmonicGauge_detail {
// The `detail` functions below are forward-declared to enable their independent
// testing
template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> weight_function(
    const tnsr::I<DataType, SpatialDim, Frame>& coords,
    double sigma_r) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_weight_function(
    const tnsr::I<DataType, SpatialDim, Frame>& coords,
    double sigma_r) noexcept;

double roll_on_function(double time, double t_start, double sigma_t) noexcept;

double time_deriv_of_roll_on_function(double time, double t_start,
                                      double sigma_t) noexcept;

template <typename DataType>
Scalar<DataType> log_factor_metric_lapse(
    const Scalar<DataType>& lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric, double exponent) noexcept;

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
    double exponent) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_power_log_factor_metric_lapse(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_powlogfac,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi, double g_exponent,
    int exponent) noexcept;

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
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi, double g_exponent,
    int exponent) noexcept;
}  // namespace DampedHarmonicGauge_detail
}  // namespace GeneralizedHarmonic

// Wrap `wrap_spacetime_deriv_of_power_log_factor_metric_lapse` here to make its
// last argument a double, allowing for `pypp::check_with_random_values` to
// work.
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame>
wrap_spacetime_deriv_of_power_log_factor_metric_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi, const double g_exponent,
    const double d_exponent) noexcept {
  const auto exponent = static_cast<int>(d_exponent);
  tnsr::a<DataType, SpatialDim, Frame> d4_powlogfac{};
  GeneralizedHarmonic::DampedHarmonicGauge_detail::
      spacetime_deriv_of_power_log_factor_metric_lapse(
          make_not_null(&d4_powlogfac), lapse, shift, spacetime_unit_normal,
          inverse_spatial_metric, sqrt_det_spatial_metric, dt_spatial_metric,
          pi, phi, g_exponent, exponent);
  return d4_powlogfac;
}

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <typename Frame>
void test_options() noexcept {
  Options<tmpl::list<
      GeneralizedHarmonic::OptionTags::GaugeHRollOnStartTime,
      GeneralizedHarmonic::OptionTags::GaugeHRollOnTimeWindow,
      GeneralizedHarmonic::OptionTags::GaugeHSpatialWeightDecayWidth<Frame>>>
      opts("");
  opts.parse(
      "GaugeHRollOnStartT : 0.\n"
      "GaugeHRollOnTWindow : 100.\n"
      "GaugeHDecayWidth : 50.\n");
  CHECK(opts.template get<
            GeneralizedHarmonic::OptionTags::GaugeHRollOnStartTime>() == 0.);
  CHECK(opts.template get<
            GeneralizedHarmonic::OptionTags::GaugeHRollOnTimeWindow>() == 100.);
  CHECK(opts.template get<GeneralizedHarmonic::OptionTags::
                              GaugeHSpatialWeightDecayWidth<Frame>>() == 50.);
}

template <size_t SpatialDim, typename Frame, typename DataType>
void test_detail_functions(const DataType& used_for_size) noexcept {
  // weight_function
  pypp::check_with_random_values<2>(
      static_cast<Scalar<DataType> (*)(
          const tnsr::I<DataType, SpatialDim, Frame>&, const double)>(
          &::GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
              SpatialDim, Frame, DataType>),
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "weight_function",
      {{{-10., 10.}, {std::numeric_limits<double>::denorm_min(), 10.}}},
      used_for_size);
  // spacetime_deriv_of_weight_function
  pypp::check_with_random_values<2>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const tnsr::I<DataType, SpatialDim, Frame>&, const double)>(
          &::GeneralizedHarmonic::DampedHarmonicGauge_detail::
              spacetime_deriv_of_weight_function<SpatialDim, Frame, DataType>),
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "spacetime_deriv_weight_function",
      {{{-10., 10.}, {std::numeric_limits<double>::denorm_min(), 10.}}},
      used_for_size);
  // roll_on_function
  pypp::check_with_random_values<1>(
      &::GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "roll_on_function", {{{std::numeric_limits<double>::denorm_min(), 10.}}},
      used_for_size);
  // time_deriv_of_roll_on_function
  pypp::check_with_random_values<1>(
      &::GeneralizedHarmonic::DampedHarmonicGauge_detail::
          time_deriv_of_roll_on_function,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "time_deriv_roll_on_function",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size);
  // log_factor_metric_lapse
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(const Scalar<DataType>&,
                                       const Scalar<DataType>&, const double)>(
          &::GeneralizedHarmonic::DampedHarmonicGauge_detail::
              log_factor_metric_lapse<DataType>),
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "log_fac", {{{std::numeric_limits<double>::denorm_min(), 10.}}},
      used_for_size);
  // spacetime_deriv_of_log_factor_metric_lapse
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&, const Scalar<DataType>&,
          const tnsr::ii<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&, const double)>(
          &::GeneralizedHarmonic::DampedHarmonicGauge_detail::
              spacetime_deriv_of_log_factor_metric_lapse<SpatialDim, Frame,
                                                         DataType>),
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "spacetime_deriv_log_fac",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size);
  // spacetime_deriv_of_power_log_factor_metric_lapse
  pypp::check_with_random_values<1>(
      &wrap_spacetime_deriv_of_power_log_factor_metric_lapse<SpatialDim, Frame,
                                                             DataType>,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "spacetime_deriv_pow_log_fac",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size,
      1.e-11);
}

//
//  Tests of the damped harmonic gauge source function
//
// Wrap `DampedHarmonicHCompute::function` here to make its time
// argument a double, allowing for `pypp::check_with_random_values` to work.
template <size_t SpatialDim, typename Frame>
typename db::item_type<GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame>>
wrap_DampedHarmonicHCompute(
    const typename db::item_type<GeneralizedHarmonic::Tags::InitialGaugeH<
        SpatialDim, Frame>>& gauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const double t, const double t_start, const double sigma_t,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double sigma_r) noexcept {
  const Slab slab(0, t);
  const Rational frac(1);
  const Time current_time(slab, frac);
  return GeneralizedHarmonic::DampedHarmonicHCompute<
      SpatialDim, Frame>::function(gauge_h_init, lapse, shift,
                                   sqrt_det_spatial_metric, spacetime_metric,
                                   current_time, t_start, sigma_t, coords,
                                   sigma_r);
}

// Compare with Python implementation
template <size_t SpatialDim, typename Frame>
void test_damped_harmonic_h_function(const DataVector& used_for_size) noexcept {
  // H_a
  pypp::check_with_random_values<1>(
      static_cast<typename db::item_type<
          GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame>> (*)(
          const typename db::item_type<
              GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame>>&,
          const Scalar<DataVector>&,
          const tnsr::I<DataVector, SpatialDim, Frame>&,
          const Scalar<DataVector>&,
          const tnsr::aa<DataVector, SpatialDim, Frame>&, const double,
          const double, const double,
          const tnsr::I<DataVector, SpatialDim, Frame>&, const double)>(
          &wrap_DampedHarmonicHCompute<SpatialDim, Frame>),
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "damped_harmonic_gauge_source_function",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size);
}

// Test term 1/4 of gauge source function for Kerr-Schild
void test_damped_harmonic_h_function_term_1_of_4(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> rdist(0.1, 1.);
  std::uniform_real_distribution<> pdist(0.01, 1.);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Randomized 3 + 1 quantities
  std::uniform_real_distribution<> dist(0.9, 1.);
  const DataVector& used_for_size = get<0>(x);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  get<0, 0>(spatial_metric) = get<1, 1>(spatial_metric) =
      get<2, 2>(spatial_metric) = 1.;
  std::uniform_real_distribution<> dist2(0., 0.12);
  const auto dspatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      spatial_metric.get(i, j) += dspatial_metric.get(i, j);
    }
  }

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));

  // Initialize settings
  const double t_start_HI = pdist(generator) * 0.1;
  const double sigma_t_HI = pdist(generator) * 0.2;

  // Initialize initial gauge function and its roll-off function
  const auto gauge_h_init = make_with_random_values<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      make_not_null(&generator), make_not_null(&rdist), x);
  const double roll_on_HI =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_HI, sigma_t_HI);

  // local H_a
  auto gauge_h_expected = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    gauge_h_expected.get(a) = (1. - roll_on_HI) * gauge_h_init.get(a);
  }

  // Check that locally computed H_a matches the returned one
  typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>
      gauge_h{};
  GeneralizedHarmonic::damped_harmonic_h<SpatialDim, Frame::Inertial>(
      make_not_null(&gauge_h), gauge_h_init, lapse, shift,
      sqrt_det_spatial_metric, spacetime_metric, t, x, 0., 0., 0., 0, 0,
      0,  // exponents
      // roll on function parameters for lapse / shift terms
      t_start_HI, sigma_t_HI, -1., -1., -1., -1., -1., -1.,
      -1. /* weight function */);

  CHECK_ITERABLE_APPROX(gauge_h_expected, gauge_h);
}

// Test term 2/4 of gauge source function for Kerr-Schild
void test_damped_harmonic_h_function_term_2_of_4(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.01, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Randomized 3 + 1 quantities
  std::uniform_real_distribution<> dist(0.9, 1.);
  const DataVector& used_for_size = get<0>(x);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  get<0, 0>(spatial_metric) = get<1, 1>(spatial_metric) =
      get<2, 2>(spatial_metric) = 1.;
  std::uniform_real_distribution<> dist2(0., 0.12);
  const auto dspatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      spatial_metric.get(i, j) += dspatial_metric.get(i, j);
    }
  }

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));

  // Initialize settings
  const double amp_coef_L1 = pdist(generator);
  const int exp_L1 = idist(generator);
  // roll on function parameters for lapse / shift terms
  const double t_start_L1 = pdist(generator) * 0.1;
  const double sigma_t_L1 = pdist(generator) * 0.2;
  const double r_max = pdist(generator) * 0.7;

  const auto& log_fac_1 = log(get(sqrt_det_spatial_metric) / get(lapse));
  const double roll_on_L1 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_L1, sigma_t_L1);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);

  const auto h_prefac1 = amp_coef_L1 * roll_on_L1 * get(weight) *
                         pow(log_fac_1, exp_L1) * log_fac_1;

  // local H_a
  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  auto gauge_h_expected = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    gauge_h_expected.get(a) = h_prefac1 * spacetime_unit_normal_one_form.get(a);
  }

  // Check that locally computed H_a matches the returned one
  typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>
      gauge_h{};
  GeneralizedHarmonic::damped_harmonic_h<SpatialDim, Frame::Inertial>(
      make_not_null(&gauge_h), gauge_h_init, lapse, shift,
      sqrt_det_spatial_metric, spacetime_metric, t, x, amp_coef_L1, 0., 0.,
      exp_L1, 0, 0,  // exponents
      // roll on function parameters for lapse / shift terms
      -1., -1., t_start_L1, sigma_t_L1, -1., -1., -1., -1.,
      r_max /* weight function */);

  CHECK_ITERABLE_APPROX(gauge_h_expected, gauge_h);
}

// Test term 3/4 of gauge source function for Kerr-Schild
void test_damped_harmonic_h_function_term_3_of_4(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.01, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Randomized 3 + 1 quantities
  std::uniform_real_distribution<> dist(0.9, 1.);
  const DataVector& used_for_size = get<0>(x);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  get<0, 0>(spatial_metric) = get<1, 1>(spatial_metric) =
      get<2, 2>(spatial_metric) = 1.;
  std::uniform_real_distribution<> dist2(0., 0.12);
  const auto dspatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      spatial_metric.get(i, j) += dspatial_metric.get(i, j);
    }
  }

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));

  // Initialize settings
  const double amp_coef_L2 = pdist(generator);
  const int exp_L2 = idist(generator);
  const double t_start_L2 = pdist(generator) * 0.1;
  const double sigma_t_L2 = pdist(generator) * 0.2;
  const double r_max = pdist(generator) * 0.7;

  const double roll_on_L2 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_L2, sigma_t_L2);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);

  const auto& log_fac_2 = log(1. / get(lapse));
  const auto h_prefac1 = amp_coef_L2 * roll_on_L2 * get(weight) *
                         pow(log_fac_2, exp_L2) * log_fac_2;

  // local H_a
  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  auto gauge_h_expected = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    gauge_h_expected.get(a) = h_prefac1 * spacetime_unit_normal_one_form.get(a);
  }

  // Check that locally computed H_a matches the returned one
  typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>
      gauge_h{};
  GeneralizedHarmonic::damped_harmonic_h<SpatialDim, Frame::Inertial>(
      make_not_null(&gauge_h), gauge_h_init, lapse, shift,
      sqrt_det_spatial_metric, spacetime_metric, t, x, 0., amp_coef_L2, 0., 0,
      exp_L2, 0,  // exponents
      // roll on function parameters for lapse / shift terms
      -1., -1., -1., -1., t_start_L2, sigma_t_L2, -1., -1.,
      r_max /* weight function */);

  CHECK_ITERABLE_APPROX(gauge_h_expected, gauge_h);
}

// Test term 4/4 of gauge source function for Kerr-Schild
void test_damped_harmonic_h_function_term_4_of_4(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.1, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Randomized 3 + 1 quantities
  std::uniform_real_distribution<> dist(0.9, 1.);
  const DataVector& used_for_size = get<0>(x);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  get<0, 0>(spatial_metric) = get<1, 1>(spatial_metric) =
      get<2, 2>(spatial_metric) = 1.;
  std::uniform_real_distribution<> dist2(0., 0.12);
  const auto dspatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      spatial_metric.get(i, j) += dspatial_metric.get(i, j);
    }
  }

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));
  const auto& one_over_lapse = 1. / get(lapse);

  // Initialize settings
  const double amp_coef_S = pdist(generator);
  const int exp_S = idist(generator);
  const double t_start_S = pdist(generator) * 0.1;
  const double sigma_t_S = pdist(generator) * 0.2;
  const double r_max = pdist(generator) * 0.7;

  const double roll_on_S =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_S, sigma_t_S);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);

  const auto& log_fac_1 = log(get(sqrt_det_spatial_metric) * one_over_lapse);
  const auto h_prefac2 = -amp_coef_S * roll_on_S * get(weight) *
                         pow(log_fac_1, exp_S) * one_over_lapse;

  // local H_a
  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  auto gauge_h_expected = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      gauge_h_expected.get(a) +=
          h_prefac2 * spacetime_metric.get(a, i + 1) * shift.get(i);
    }
  }

  // Check that locally computed H_a match returned one
  typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>
      gauge_h{};
  GeneralizedHarmonic::damped_harmonic_h<SpatialDim, Frame::Inertial>(
      make_not_null(&gauge_h), gauge_h_init, lapse, shift,
      sqrt_det_spatial_metric, spacetime_metric, t, x, 0., 0., amp_coef_S, 0, 0,
      exp_S,  // exponents
      // roll on function parameters for lapse / shift terms
      -1., -1., -1., -1., -1., -1., t_start_S, sigma_t_S,
      r_max /* weight function */);

  CHECK_ITERABLE_APPROX(gauge_h_expected, gauge_h);
}

// The next three functions test the gauge source function for
// Schwarzschild spacetime, in Kerr-Schild coordinates.

// Test term 2/4 of gauge source function for Schwarzschild in Kerr-Schild coord
template <typename Solution>
void test_damped_harmonic_h_function_term_2_of_4_analytic_schwarzschild(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.01, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& shift = get<gr::Tags::Shift<SpatialDim>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<SpatialDim>>(vars);

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));

  // Initialize settings
  const double amp_coef_L1 = pdist(generator);
  const int exp_L1 = idist(generator);
  // roll on function parameters for lapse / shift terms
  const double t_start_L1 = pdist(generator) * 0.1;
  const double sigma_t_L1 = pdist(generator) * 0.2;
  const double r_max = pdist(generator) * 0.7;
  const double roll_on_L1 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_L1, sigma_t_L1);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);

  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>
      gauge_h{};
  GeneralizedHarmonic::damped_harmonic_h<SpatialDim, Frame::Inertial>(
      make_not_null(&gauge_h), gauge_h_init, lapse, shift,
      sqrt_det_spatial_metric, spacetime_metric, t, x, amp_coef_L1, 0., 0.,
      exp_L1, 0, 0,  // exponents
      // roll on function parameters for lapse / shift terms
      -1., -1., t_start_L1, sigma_t_L1, -1., -1., -1., -1.,
      r_max /* weight function */);

  // Evaluate analytic solution for Schwarzschild, ab initio.
  const double mass = solution.mass();
  const std::array<double, SpatialDim> center = solution.center();
  // 1) shift coords with BH at the center
  auto x_minus_center = x;
  // DataVector r_squared(get_size(x), 0.);
  auto r_squared = get<0>(x);
  r_squared -= get<0>(x);
  for (size_t i = 0; i < SpatialDim; ++i) {
    x_minus_center.get(i) -= gsl::at(center, i);
    r_squared += square(x_minus_center.get(i));
  }
  const DataVector r_ = sqrt(r_squared);
  // 2) compute scalar function that defines KerrSchild coords
  const DataVector H = mass * (1. / r_);
  const DataVector H2 = H * 2.;
  // 3) compute lapse
  const Scalar<DataVector> lapse_ab_initio{sqrt(1. / (1. + H2))};
  // 4) compute 3-metric
  auto spatial_metric_ab_initio =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      spatial_metric_ab_initio.get(i, j) =
          H2 * x_minus_center.get(i) * x_minus_center.get(j) / r_squared;
      if (i == j) {
        spatial_metric_ab_initio.get(i, i) += 1.;
      }
    }
  }
  const auto& spacetime_unit_normal_one_form_ab_initio =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio);
  const auto& det_spatial_metric_ab_initio =
      determinant_and_inverse(spatial_metric_ab_initio).first;
  Scalar<DataVector> sqrt_det_spatial_metric_ab_initio(
      sqrt(get(det_spatial_metric_ab_initio)));

  // compute GR dependent terms
  const auto& log_fac_1 =
      log(get(sqrt_det_spatial_metric_ab_initio) / get(lapse_ab_initio));

  const auto h_prefac1 = amp_coef_L1 * roll_on_L1 * get(weight) *
                         pow(log_fac_1, exp_L1) * log_fac_1;

  // local H_a
  auto gauge_h_expected = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    gauge_h_expected.get(a) =
        h_prefac1 * spacetime_unit_normal_one_form_ab_initio.get(a);
  }

  // Check that locally computed H_a matches the returned one
  CHECK_ITERABLE_APPROX(gauge_h_expected, gauge_h);
}

// Test term 3/4 of gauge source function for Schwarzschild in Kerr-Schild coord
template <typename Solution>
void test_damped_harmonic_h_function_term_3_of_4_analytic_schwarzschild(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.01, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& shift = get<gr::Tags::Shift<SpatialDim>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<SpatialDim>>(vars);

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));

  // Initialize settings
  const double amp_coef_L2 = pdist(generator);
  const int exp_L2 = idist(generator);
  const double t_start_L2 = pdist(generator) * 0.1;
  const double sigma_t_L2 = pdist(generator) * 0.2;
  const double r_max = pdist(generator) * 0.7;
  const double roll_on_L2 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_L2, sigma_t_L2);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);

  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>
      gauge_h{};
  GeneralizedHarmonic::damped_harmonic_h<SpatialDim, Frame::Inertial>(
      make_not_null(&gauge_h), gauge_h_init, lapse, shift,
      sqrt_det_spatial_metric, spacetime_metric, t, x, 0., amp_coef_L2, 0., 0,
      exp_L2, 0,  // exponents
      // roll on function parameters for lapse / shift terms
      -1., -1., -1., -1., t_start_L2, sigma_t_L2, -1., -1.,
      r_max /* weight function */);

  // Evaluate analytic solution for Schwarzschild, ab initio.
  const double mass = solution.mass();
  const std::array<double, SpatialDim> center = solution.center();
  // 1) shift coords with BH at the center
  auto x_minus_center = x;
  // DataVector r_squared(get_size(x), 0.);
  auto r_squared = get<0>(x);
  r_squared -= get<0>(x);
  for (size_t i = 0; i < SpatialDim; ++i) {
    x_minus_center.get(i) -= gsl::at(center, i);
    r_squared += square(x_minus_center.get(i));
  }
  const DataVector r_ = sqrt(r_squared);
  // 2) compute scalar function that defines KerrSchild coords
  const DataVector H = mass * (1. / r_);
  const DataVector H2 = H * 2.;
  // 3) compute lapse
  const Scalar<DataVector> lapse_ab_initio{sqrt(1. / (1. + H2))};

  const auto& spacetime_unit_normal_one_form_ab_initio =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio);

  // compute GR dependent terms
  const auto& log_fac_2 = log(1. / get(lapse_ab_initio));

  // compute other terms
  const auto h_prefac1 = amp_coef_L2 * roll_on_L2 * get(weight) *
                         pow(log_fac_2, exp_L2) * log_fac_2;

  // local H_a
  auto gauge_h_expected = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    gauge_h_expected.get(a) =
        h_prefac1 * spacetime_unit_normal_one_form_ab_initio.get(a);
  }

  // Check that locally computed H_a matches the returned one
  CHECK_ITERABLE_APPROX(gauge_h_expected, gauge_h);
}

// Test term 4/4 of gauge source function for Schwarzschild in Kerr-Schild coord
template <typename Solution>
void test_damped_harmonic_h_function_term_4_of_4_analytic_schwarzschild(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.01, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& shift = get<gr::Tags::Shift<SpatialDim>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<SpatialDim>>(vars);

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));

  // Initialize settings
  const double amp_coef_S = pdist(generator);
  const int exp_S = idist(generator);
  const double t_start_S = pdist(generator) * 0.1;
  const double sigma_t_S = pdist(generator) * 0.2;
  const double r_max = pdist(generator) * 0.7;
  const double roll_on_S =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_S, sigma_t_S);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);

  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>
      gauge_h{};
  GeneralizedHarmonic::damped_harmonic_h<SpatialDim, Frame::Inertial>(
      make_not_null(&gauge_h), gauge_h_init, lapse, shift,
      sqrt_det_spatial_metric, spacetime_metric, t, x, 0., 0., amp_coef_S, 0, 0,
      exp_S,  // exponents
      // roll on function parameters for lapse / shift terms
      -1., -1., -1., -1., -1., -1., t_start_S, sigma_t_S,
      r_max /* weight function */);

  // Evaluate analytic solution for Schwarzschild, ab initio.
  const double mass = solution.mass();
  const std::array<double, SpatialDim> center = solution.center();
  // 1) shift coords with BH at the center
  auto x_minus_center = x;
  // DataVector r_squared(get_size(x), 0.);
  auto r_squared = get<0>(x);
  r_squared -= get<0>(x);
  for (size_t i = 0; i < SpatialDim; ++i) {
    x_minus_center.get(i) -= gsl::at(center, i);
    r_squared += square(x_minus_center.get(i));
  }
  const DataVector r_ = sqrt(r_squared);
  // 2) compute scalar function that defines KerrSchild coords
  const DataVector H = mass * (1. / r_);
  const DataVector H2 = H * 2.;
  // 3) compute lapse
  const Scalar<DataVector> lapse_ab_initio{sqrt(1. / (1. + H2))};
  // 4) compute shift
  auto shift_ab_initio =
      make_with_value<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    shift_ab_initio.get(i) = (H2 / (1. + H2)) * x_minus_center.get(i) / r_;
  }
  // 5) compute 3-metric
  auto spatial_metric_ab_initio =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      spatial_metric_ab_initio.get(i, j) =
          H2 * x_minus_center.get(i) * x_minus_center.get(j) / r_squared;
      if (i == j) {
        spatial_metric_ab_initio.get(i, i) += 1.;
      }
    }
  }
  const auto& spacetime_metric_ab_initio = gr::spacetime_metric(
      lapse_ab_initio, shift_ab_initio, spatial_metric_ab_initio);
  const auto& spacetime_unit_normal_one_form_ab_initio =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio);
  const auto& det_spatial_metric_ab_initio =
      determinant_and_inverse(spatial_metric_ab_initio).first;
  Scalar<DataVector> sqrt_det_spatial_metric_ab_initio(
      sqrt(get(det_spatial_metric_ab_initio)));

  // compute GR dependent terms
  const auto& log_fac_1 =
      log(get(sqrt_det_spatial_metric_ab_initio) / get(lapse_ab_initio));

  const auto h_prefac2 = -amp_coef_S * roll_on_S * get(weight) *
                         pow(log_fac_1, exp_S) / get(lapse_ab_initio);

  // local H_a
  auto gauge_h_expected = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::GaugeH<SpatialDim, Frame::Inertial>>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      gauge_h_expected.get(a) += h_prefac2 *
                                 spacetime_metric_ab_initio.get(a, i + 1) *
                                 shift_ab_initio.get(i);
    }
  }

  // Check that locally computed H_a match returned one
  CHECK_ITERABLE_APPROX(gauge_h_expected, gauge_h);
}

//
//  Tests of spacetime derivatives of the damped harmonic gauge source function
//
// Wrap `SpacetimeDerivDampedHarmonicHCompute::function` here to make its time
// argument a double, allowing for `pypp::check_with_random_values` to work.
template <size_t SpatialDim, typename Frame>
typename db::item_type<
    GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<SpatialDim, Frame>>
wrap_SpacetimeDerivDampedHarmonicHCompute(
    const typename db::item_type<GeneralizedHarmonic::Tags::InitialGaugeH<
        SpatialDim, Frame>>& gauge_h_init,
    const typename db::item_type<
        GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<
            SpatialDim, Frame>>& dgauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const tnsr::a<DataVector, SpatialDim, Frame>&
        spacetime_unit_normal_one_form,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, const double t,
    const double t_start, const double sigma_t,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double sigma_r) noexcept {
  const Slab slab(0, t);
  const Rational frac(1);
  const Time current_time(slab, frac);
  return GeneralizedHarmonic::SpacetimeDerivDampedHarmonicHCompute<
      SpatialDim, Frame>::function(gauge_h_init, dgauge_h_init, lapse, shift,
                                   spacetime_unit_normal_one_form,
                                   sqrt_det_spatial_metric,
                                   inverse_spatial_metric, spacetime_metric, pi,
                                   phi, current_time, t_start, sigma_t, coords,
                                   sigma_r);
}
// Compare with Python implementation
template <size_t SpatialDim, typename Frame>
void test_deriv_damped_harmonic_h_function(
    const DataVector& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<typename db::item_type<
          GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<SpatialDim,
                                                          Frame>> (*)(
          const typename db::item_type<
              GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame>>&,
          const typename db::item_type<
              GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<SpatialDim,
                                                                     Frame>>&,
          const Scalar<DataVector>&,
          const tnsr::I<DataVector, SpatialDim, Frame>&,
          const tnsr::a<DataVector, SpatialDim, Frame>&,
          const Scalar<DataVector>&,
          const tnsr::II<DataVector, SpatialDim, Frame>&,
          const tnsr::aa<DataVector, SpatialDim, Frame>&,
          const tnsr::aa<DataVector, SpatialDim, Frame>&,
          const tnsr::iaa<DataVector, SpatialDim, Frame>&, const double,
          const double, const double,
          const tnsr::I<DataVector, SpatialDim, Frame>&, const double)>(
          &wrap_SpacetimeDerivDampedHarmonicHCompute<SpatialDim, Frame>),
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "spacetime_deriv_damped_harmonic_gauge_source_function", {{{0.1, 10.}}},
      used_for_size, 1.e-11);
}

// Test term 1/4 of spacetime deriv of gauge source function
void test_deriv_damped_harmonic_h_function_term_1_of_4(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> rdist(0.1, 1.);
  std::uniform_real_distribution<> pdist(0.01, 1.);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Randomized 3 + 1 quantities
  // Note: Ranges from which random numbers are drawn to populate 3+1 tensors
  //       below were chosen to not throw FPEs (by trial & error).
  std::uniform_real_distribution<> dist(0.9, 1.);
  const DataVector& used_for_size = get<0>(x);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto dt_lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto d_lapse =
      make_with_random_values<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto dt_shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto d_shift = make_with_random_values<
      tnsr::iJ<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  get<0, 0>(spatial_metric) = get<1, 1>(spatial_metric) =
      get<2, 2>(spatial_metric) = 1.;
  std::uniform_real_distribution<> dist2(0., 0.12);
  const auto dspatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      spatial_metric.get(i, j) += dspatial_metric.get(i, j);
    }
  }
  const auto dt_spatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  const auto d_spatial_metric = make_with_random_values<
      tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));
  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);

  // Initialize settings
  // roll on function parameters
  const double t_start_HI = pdist(generator) * 0.1;
  const double sigma_t_HI = pdist(generator) * 0.2;

  // Tempering functions
  const double roll_on_HI =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_HI, sigma_t_HI);
  const auto d0_roll_on_HI = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      time_deriv_of_roll_on_function(t, t_start_HI, sigma_t_HI);

  // Initialize initial gauge function and its roll-off function
  const auto gauge_h_init = make_with_random_values<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      make_not_null(&generator), make_not_null(&rdist), x);
  const auto d4_gauge_h_init = make_with_random_values<typename db::item_type<
      GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<SpatialDim,
                                                             Frame::Inertial>>>(
      make_not_null(&generator), make_not_null(&rdist), x);

  // Calc \f$ \partial_a T1 \f$
  auto dT1 =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t b = 0; b < SpatialDim + 1; ++b) {
    dT1.get(0, b) -= gauge_h_init.get(b) * d0_roll_on_HI;
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      dT1.get(a, b) += (1. - roll_on_HI) * d4_gauge_h_init.get(a, b);
    }
  }

  // Calc \f$ \partial_a H_b = dT1_{ab} \f$
  const auto& d4_gauge_h_expected = dT1;

  // Check that locally computed \f$\partial_a H_b\f$ matches returned one
  typename db::item_type<GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<
      SpatialDim, Frame::Inertial>>
      d4_gauge_h{};
  GeneralizedHarmonic::spacetime_deriv_damped_harmonic_h<SpatialDim,
                                                         Frame::Inertial>(
      make_not_null(&d4_gauge_h), gauge_h_init, d4_gauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, t, x, 0., 0., 0., 0, 0,
      0,  // exponents
      // roll on function parameters
      t_start_HI, sigma_t_HI, -1., -1., -1., -1., -1., -1.,
      -1. /* weight function */);

  CHECK_ITERABLE_APPROX(d4_gauge_h_expected, d4_gauge_h);
}

// Test term 2/4 of spacetime deriv of gauge source function
void test_deriv_damped_harmonic_h_function_term_2_of_4(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.01, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Randomized 3 + 1 quantities
  // Note: Ranges from which random numbers are drawn to populate 3+1 tensors
  //       below were chosen to not throw FPEs (by trial & error).
  std::uniform_real_distribution<> dist(0.9, 1.);
  const DataVector& used_for_size = get<0>(x);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto dt_lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto d_lapse =
      make_with_random_values<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto dt_shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto d_shift = make_with_random_values<
      tnsr::iJ<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  get<0, 0>(spatial_metric) = get<1, 1>(spatial_metric) =
      get<2, 2>(spatial_metric) = 1.;
  std::uniform_real_distribution<> dist2(0., 0.12);
  const auto dspatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      spatial_metric.get(i, j) += dspatial_metric.get(i, j);
    }
  }
  const auto dt_spatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  const auto d_spatial_metric = make_with_random_values<
      tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto spacetime_unit_normal =
      gr::spacetime_normal_vector<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift);
  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift, inverse_spatial_metric);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));
  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);

  // commonly used terms
  const auto exp_fac_1 = 1. / 2.;
  const auto& log_fac_1 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::log_factor_metric_lapse<
          DataVector>(lapse, sqrt_det_spatial_metric, exp_fac_1);

  // Initialize settings
  const double amp_coef_L1 = pdist(generator);
  const int exp_L1 = idist(generator);
  const double t_start_L1 = pdist(generator) * 0.1;
  const double sigma_t_L1 = pdist(generator) * 0.2;
  const double r_max = pdist(generator) * 0.7;

  // Tempering functions
  const double roll_on_L1 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_L1, sigma_t_L1);
  const auto d0_roll_on_L1 = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      time_deriv_of_roll_on_function(t, t_start_L1, sigma_t_L1);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);
  auto d4_weight = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      spacetime_deriv_of_weight_function<SpatialDim, Frame::Inertial,
                                         DataVector>(x, r_max);

  // coeffs that enter gauge source function
  const auto mu_L1 =
      amp_coef_L1 * roll_on_L1 * get(weight) * pow(get(log_fac_1), exp_L1);

  // Calc \f$ \mu_1 = \mu_{L1} log(rootg/N) = R W log(rootg/N)^5\f$
  const auto mu1 = mu_L1 * get(log_fac_1);

  // Calc \f$ \partial_a [R W] \f$
  auto d4_RW_L1 = d4_weight;
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_RW_L1.get(a) *= roll_on_L1;
  }
  d4_RW_L1.get(0) += get(weight) * d0_roll_on_L1;

  // Calc derivs of \f$ \mu_1 \f$
  auto d4_mu1 =
      make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse), 0.);
  {
    const auto& d4_log_fac_mu1 =
        GeneralizedHarmonic::DampedHarmonicGauge_detail ::
            spacetime_deriv_of_power_log_factor_metric_lapse<
                SpatialDim, Frame::Inertial, DataVector>(
                lapse, shift, spacetime_unit_normal, inverse_spatial_metric,
                sqrt_det_spatial_metric, dt_spatial_metric, pi, phi, exp_fac_1,
                exp_L1 + 1);
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      // \f$ \partial_a \mu_1 \f$
      d4_mu1.get(a) =
          amp_coef_L1 * pow(get(log_fac_1), exp_L1 + 1) * d4_RW_L1.get(a) +
          amp_coef_L1 * roll_on_L1 * get(weight) * d4_log_fac_mu1.get(a);
    }
  }

  // Calc \f$ \partial_a N = {\partial_0 N, \partial_i N} \f$
  auto d4_N = make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
      get(lapse), 0.);
  d4_N.get(0) = get(dt_lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_N.get(1 + i) = d_lapse.get(i);
  }

  // Calc \f$ \partial_a n_b = {-\partial_a N, 0, 0, 0} \f$
  auto d4_normal_one_form =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse), 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_normal_one_form.get(a, 0) -= d4_N.get(a);
  }

  // Calc \f$ \partial_a T2 \f$
  auto dT2 =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      dT2.get(a, b) += mu1 * d4_normal_one_form.get(a, b) +
                       d4_mu1.get(a) * spacetime_unit_normal_one_form.get(b);
    }
  }

  // Initialize initial gauge function and its roll-off function
  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  const auto d4_gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<SpatialDim,
                                                             Frame::Inertial>>>(
      x, 0.);

  // Calc \f$ \partial_a H_b = dT2_{ab} \f$
  const auto& d4_gauge_h_expected = dT2;

  // Check that locally computed \f$\partial_a H_b\f$ matches returned one
  typename db::item_type<GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<
      SpatialDim, Frame::Inertial>>
      d4_gauge_h{};
  GeneralizedHarmonic::spacetime_deriv_damped_harmonic_h<SpatialDim,
                                                         Frame::Inertial>(
      make_not_null(&d4_gauge_h), gauge_h_init, d4_gauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, t, x, amp_coef_L1, 0.,
      0., exp_L1, 0, 0,  // exponents
      // roll on function parameters
      -1., -1., t_start_L1, sigma_t_L1, -1., -1., -1., -1.,
      r_max /* weight function */);

  CHECK_ITERABLE_APPROX(d4_gauge_h_expected, d4_gauge_h);
}

// Test term 3/4 of spacetime deriv of gauge source function
void test_deriv_damped_harmonic_h_function_term_3_of_4(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.01, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Randomized 3 + 1 quantities
  // Note: Ranges from which random numbers are drawn to populate 3+1 tensors
  //       below were chosen to not throw FPEs (by trial & error).
  std::uniform_real_distribution<> dist(0.9, 1.);
  const DataVector& used_for_size = get<0>(x);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto dt_lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto d_lapse =
      make_with_random_values<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto dt_shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto d_shift = make_with_random_values<
      tnsr::iJ<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  get<0, 0>(spatial_metric) = get<1, 1>(spatial_metric) =
      get<2, 2>(spatial_metric) = 1.;
  std::uniform_real_distribution<> dist2(0., 0.12);
  const auto dspatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      spatial_metric.get(i, j) += dspatial_metric.get(i, j);
    }
  }
  const auto dt_spatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  const auto d_spatial_metric = make_with_random_values<
      tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto spacetime_unit_normal =
      gr::spacetime_normal_vector<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift);
  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift, inverse_spatial_metric);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));
  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);

  // commonly used terms
  const auto exp_fac_2 = 0.;
  const auto& log_fac_2 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::log_factor_metric_lapse<
          DataVector>(lapse, sqrt_det_spatial_metric, exp_fac_2);

  // Initialize settings
  const double amp_coef_L2 = pdist(generator);
  const int exp_L2 = idist(generator);
  const double t_start_L2 = pdist(generator) * 0.1;
  const double sigma_t_L2 = pdist(generator) * 0.2;
  // weight function
  const double r_max = pdist(generator) * 0.7;

  // Tempering functions
  const double roll_on_L2 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_L2, sigma_t_L2);
  const auto d0_roll_on_L2 = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      time_deriv_of_roll_on_function(t, t_start_L2, sigma_t_L2);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);
  auto d4_weight = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      spacetime_deriv_of_weight_function<SpatialDim, Frame::Inertial,
                                         DataVector>(x, r_max);

  // coeffs that enter gauge source function
  const auto mu_L2 =
      amp_coef_L2 * roll_on_L2 * get(weight) * pow(get(log_fac_2), exp_L2);

  // Calc \f$ \mu_2 = \mu_{L2} log(1/N) = R W log(1/N)^5\f$
  const auto mu2 = mu_L2 * get(log_fac_2);

  // Calc \f$ \partial_a [R W] \f$
  auto d4_RW_L2 = d4_weight;
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_RW_L2.get(a) *= roll_on_L2;
  }
  d4_RW_L2.get(0) += get(weight) * d0_roll_on_L2;

  // Calc derivs of \f$ \mu_2 \f$
  auto d4_mu2 =
      make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse), 0.);
  {
    const auto& d4_log_fac_mu2 =
        GeneralizedHarmonic::DampedHarmonicGauge_detail ::
            spacetime_deriv_of_power_log_factor_metric_lapse<
                SpatialDim, Frame::Inertial, DataVector>(
                lapse, shift, spacetime_unit_normal, inverse_spatial_metric,
                sqrt_det_spatial_metric, dt_spatial_metric, pi, phi, exp_fac_2,
                exp_L2 + 1);
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      // \f$ \partial_a \mu_2 \f$
      d4_mu2.get(a) =
          amp_coef_L2 * pow(get(log_fac_2), exp_L2 + 1) * d4_RW_L2.get(a) +
          amp_coef_L2 * roll_on_L2 * get(weight) * d4_log_fac_mu2.get(a);
    }
  }

  // Calc \f$ \partial_a N = {\partial_0 N, \partial_i N} \f$
  auto d4_N = make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
      get(lapse), 0.);
  d4_N.get(0) = get(dt_lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_N.get(1 + i) = d_lapse.get(i);
  }

  // Calc \f$ \partial_a n_b = {-\partial_a N, 0, 0, 0} \f$
  auto d4_normal_one_form =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse), 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_normal_one_form.get(a, 0) -= d4_N.get(a);
  }

  // Calc \f$ \partial_a T2 \f$
  auto dT2 =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      dT2.get(a, b) += mu2 * d4_normal_one_form.get(a, b) +
                       d4_mu2.get(a) * spacetime_unit_normal_one_form.get(b);
    }
  }

  // Initialize initial gauge function and its roll-off function
  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  const auto d4_gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<SpatialDim,
                                                             Frame::Inertial>>>(
      x, 0.);

  // Calc \f$ \partial_a H_b = dT2_{ab} \f$
  const auto& d4_gauge_h_expected = dT2;

  // Check that locally computed \f$\partial_a H_b\f$ matches returned one
  typename db::item_type<GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<
      SpatialDim, Frame::Inertial>>
      d4_gauge_h{};
  GeneralizedHarmonic::spacetime_deriv_damped_harmonic_h<SpatialDim,
                                                         Frame::Inertial>(
      make_not_null(&d4_gauge_h), gauge_h_init, d4_gauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, t, x, 0., amp_coef_L2,
      0., 0, exp_L2, 0,  // exponents
      // roll on function parameters for lapse / shift terms
      -1., -1., -1., -1., t_start_L2, sigma_t_L2, -1., -1.,
      r_max /* weight function */);

  CHECK_ITERABLE_APPROX(d4_gauge_h_expected, d4_gauge_h);
}

// Test term 4/4 of spacetime deriv of gauge source function
void test_deriv_damped_harmonic_h_function_term_4_of_4(
    const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.01, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Randomized 3 + 1 quantities
  // Note: Ranges from which random numbers are drawn to populate 3+1 tensors
  //       below were chosen to not throw FPEs (by trial & error).
  std::uniform_real_distribution<> dist(0.9, 1.);
  const DataVector& used_for_size = get<0>(x);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto dt_lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto d_lapse =
      make_with_random_values<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto dt_shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto d_shift = make_with_random_values<
      tnsr::iJ<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  get<0, 0>(spatial_metric) = get<1, 1>(spatial_metric) =
      get<2, 2>(spatial_metric) = 1.;
  std::uniform_real_distribution<> dist2(0., 0.12);
  const auto dspatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      spatial_metric.get(i, j) += dspatial_metric.get(i, j);
    }
  }
  const auto dt_spatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  const auto d_spatial_metric = make_with_random_values<
      tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto spacetime_unit_normal =
      gr::spacetime_normal_vector<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift);
  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift, inverse_spatial_metric);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));
  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);
  const auto d4_spacetime_metric = gr::derivatives_of_spacetime_metric(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, spatial_metric,
      dt_spatial_metric, d_spatial_metric);

  // commonly used terms
  const auto exp_fac_1 = 1. / 2.;
  const auto& one_over_lapse = 1. / get(lapse);
  const auto& log_fac_1 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::log_factor_metric_lapse<
          DataVector>(lapse, sqrt_det_spatial_metric, exp_fac_1);

  // Initialize settings
  const double amp_coef_S = pdist(generator);
  const int exp_S = idist(generator);
  const double t_start_S = pdist(generator) * 0.1;
  const double sigma_t_S = pdist(generator) * 0.2;
  const double r_max = pdist(generator) * 0.7;

  // Tempering functions
  const double roll_on_S =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_S, sigma_t_S);
  const auto d0_roll_on_S = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      time_deriv_of_roll_on_function(t, t_start_S, sigma_t_S);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);
  auto d4_weight = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      spacetime_deriv_of_weight_function<SpatialDim, Frame::Inertial,
                                         DataVector>(x, r_max);

  // coeffs that enter gauge source function
  const auto mu_S =
      amp_coef_S * roll_on_S * get(weight) * pow(get(log_fac_1), exp_S);
  const auto mu_S_over_N = mu_S * one_over_lapse;

  // Calc \f$ \partial_a [R W] \f$
  auto d4_RW_S = d4_weight;
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_RW_S.get(a) *= roll_on_S;
  }
  d4_RW_S.get(0) += get(weight) * d0_roll_on_S;

  // Calc derivs of \f$ \mu_S \f$
  auto d4_mu_S =
      make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse), 0.);
  {
    const auto& d4_log_fac_muS =
        GeneralizedHarmonic::DampedHarmonicGauge_detail ::
            spacetime_deriv_of_power_log_factor_metric_lapse<
                SpatialDim, Frame::Inertial, DataVector>(
                lapse, shift, spacetime_unit_normal, inverse_spatial_metric,
                sqrt_det_spatial_metric, dt_spatial_metric, pi, phi, exp_fac_1,
                exp_S);
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      // \f$ \partial_a \mu_{L1} \f$
      d4_mu_S.get(a) =
          amp_coef_S * d4_RW_S.get(a) * pow(get(log_fac_1), exp_S) +
          amp_coef_S * roll_on_S * get(weight) * d4_log_fac_muS.get(a);
    }
  }

  // Calc \f$ \partial_a N = {\partial_0 N, \partial_i N} \f$
  auto d4_N = make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
      get(lapse), 0.);
  d4_N.get(0) = get(dt_lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_N.get(1 + i) = d_lapse.get(i);
  }

  // Calc \f$ \partial_a N^i = {\partial_0 N^i, \partial_j N^i} \f$
  auto d4_shift =
      make_with_value<tnsr::aB<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse), 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_shift.get(0, 1 + i) = dt_shift.get(i);
    for (size_t j = 0; j < SpatialDim; ++j) {
      d4_shift.get(1 + j, 1 + i) = d_shift.get(j, i);
    }
  }

  // \f[ \partial_a (\mu_S/N) = (1/N) \partial_a \mu_{S}
  //         - (\mu_{S}/N^2) \partial_a N
  // \f]
  auto d4_muS_over_N =
      make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse), 0.);
  {
    auto prefac = -mu_S * one_over_lapse * one_over_lapse;
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_muS_over_N.get(a) +=
          one_over_lapse * d4_mu_S.get(a) + prefac * d4_N.get(a);
    }
  }

  // Calc \f$ \partial_a T3 \f$ (note minus sign)
  auto dT3 =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        dT3.get(a, b) -= d4_muS_over_N.get(a) * spacetime_metric.get(b, i + 1) *
                         shift.get(i);
        dT3.get(a, b) -=
            mu_S_over_N * d4_spacetime_metric.get(a, b, i + 1) * shift.get(i);
        dT3.get(a, b) -= mu_S_over_N * spacetime_metric.get(b, i + 1) *
                         d4_shift.get(a, i + 1);
      }
    }
  }

  // Initialize initial gauge function and its roll-off function
  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  const auto d4_gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<SpatialDim,
                                                             Frame::Inertial>>>(
      x, 0.);

  // Calc \f$ \partial_a H_b = dT3_{ab} \f$
  const auto& d4_gauge_h_expected = dT3;

  // Check that locally computed \f$\partial_a H_b\f$ matches returned one
  typename db::item_type<GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<
      SpatialDim, Frame::Inertial>>
      d4_gauge_h{};
  GeneralizedHarmonic::spacetime_deriv_damped_harmonic_h<SpatialDim,
                                                         Frame::Inertial>(
      make_not_null(&d4_gauge_h), gauge_h_init, d4_gauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, t, x, 0., 0.,
      amp_coef_S, 0, 0, exp_S,  // exponents
      // roll on function parameters for lapse / shift terms
      -1., -1., -1., -1., -1., -1., t_start_S, sigma_t_S,
      r_max /* weight function */);

  CHECK_ITERABLE_APPROX(d4_gauge_h_expected, d4_gauge_h);
}

// The next three functions test spacetime derivatives of the gauge source
// function for Schwarzschild spacetime, in Kerr-Schild coordinates.

// Test term 2/4 of gauge source function for Schwarzschild in Kerr-Schild coord
template <typename Solution>
void test_deriv_damped_harmonic_h_function_term_2_of_4_analytic_schwarzschild(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.01, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<SpatialDim>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<SpatialDim>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<SpatialDim>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<SpatialDim>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto spacetime_unit_normal =
      gr::spacetime_normal_vector<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift);
  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift, inverse_spatial_metric);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));
  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);

  // Initialize settings
  const double amp_coef_L1 = pdist(generator);
  const int exp_L1 = idist(generator);
  const double t_start_L1 = pdist(generator) * 0.1;
  const double sigma_t_L1 = pdist(generator) * 0.2;
  const double r_max = pdist(generator) * 0.7;

  // Initialize initial gauge function and its roll-off function
  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  const auto d4_gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<SpatialDim,
                                                             Frame::Inertial>>>(
      x, 0.);

  // compute d4H using function being tested
  typename db::item_type<GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<
      SpatialDim, Frame::Inertial>>
      d4_gauge_h{};
  GeneralizedHarmonic::spacetime_deriv_damped_harmonic_h<SpatialDim,
                                                         Frame::Inertial>(
      make_not_null(&d4_gauge_h), gauge_h_init, d4_gauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, t, x, amp_coef_L1, 0.,
      0., exp_L1, 0, 0,  // exponents
      // roll on function parameters
      -1., -1., t_start_L1, sigma_t_L1, -1., -1., -1., -1.,
      r_max /* weight function */);

  // Tempering functions
  const double roll_on_L1 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_L1, sigma_t_L1);
  const auto d0_roll_on_L1 = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      time_deriv_of_roll_on_function(t, t_start_L1, sigma_t_L1);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);
  auto d4_weight = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      spacetime_deriv_of_weight_function<SpatialDim, Frame::Inertial,
                                         DataVector>(x, r_max);

  // Evaluate analytic solution for Schwarzschild, ab initio.
  const double mass = solution.mass();
  const std::array<double, SpatialDim> center = solution.center();
  // 1) shift coords with BH at the center
  auto x_minus_center = x;
  // DataVector r_squared(get_size(x), 0.);
  auto r_squared = get<0>(x);
  r_squared -= get<0>(x);
  for (size_t i = 0; i < SpatialDim; ++i) {
    x_minus_center.get(i) -= gsl::at(center, i);
    r_squared += square(x_minus_center.get(i));
  }
  const DataVector r_ = sqrt(r_squared);
  // 2) compute scalar function that defines KerrSchild coords
  const DataVector H = mass * (1. / r_);
  const DataVector H2 = H * 2.;
  // 3) compute 3D null_one_form and its derivatives
  tnsr::i<DataVector, SpatialDim, Frame::Inertial> null_one_form{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    null_one_form.get(i) = x_minus_center.get(i) / r_;
  }
  auto null_vec = null_one_form;
  tnsr::ij<DataVector, SpatialDim, Frame::Inertial> deriv_null_one_form{};
  const DataVector denom = 1. / r_squared;
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      deriv_null_one_form.get(i, j) =
          -(null_one_form.get(i) * null_one_form.get(j)) / r_;
      if (i == j) {
        deriv_null_one_form.get(j, i) += (1. / r_);
      }
    }
  }
  // 4) compute deriv of H
  tnsr::i<DataVector, SpatialDim, Frame::Inertial> deriv_H{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    deriv_H.get(i) = -(H / r_squared) * x_minus_center.get(i);
  }
  // 5) compute lapse & its derivatives
  const Scalar<DataVector> lapse_ab_initio{sqrt(1. / (1. + H2))};
  const auto dt_lapse_ab_initio = make_with_value<Scalar<DataVector>>(x, 0.);
  auto d_lapse_ab_initio =
      make_with_value<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d_lapse_ab_initio.get(i) -= (pow<3>(get(lapse_ab_initio)) * deriv_H.get(i));
  }
  // 6) compute shift & its derivatives
  auto shift_ab_initio =
      make_with_value<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    shift_ab_initio.get(i) = (H2 / (1. + H2)) * null_vec.get(i);
  }
  const auto dt_shift_ab_initio =
      make_with_value<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  auto d_shift_ab_initio =
      make_with_value<tnsr::iJ<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      d_shift_ab_initio.get(i, j) =
          2. * pow<4>(get(lapse_ab_initio)) * null_vec.get(j) * deriv_H.get(i) +
          H2 * square(get(lapse_ab_initio)) * deriv_null_one_form.get(i, j);
    }
  }
  // 7) compute 3-metric & its derivatives
  auto spatial_metric_ab_initio =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      spatial_metric_ab_initio.get(i, j) =
          H2 * null_vec.get(i) * null_vec.get(j);
      if (i == j) {
        spatial_metric_ab_initio.get(i, i) += 1.;
      }
    }
  }
  auto dt_spatial_metric_ab_initio =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  auto d_spatial_metric_ab_initio =
      make_with_value<tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>>(x,
                                                                          0.);
  for (size_t k = 0; k < SpatialDim; ++k) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = i; j < SpatialDim; ++j) {
        d_spatial_metric_ab_initio.get(k, i, j) =
            2. * null_vec.get(i) * null_vec.get(j) * deriv_H.get(k) +
            H2 * (null_one_form.get(i) * deriv_null_one_form.get(k, j) +
                  null_one_form.get(j) * deriv_null_one_form.get(k, i));
      }
    }
  }

  const auto& spacetime_metric_ab_initio = gr::spacetime_metric(
      lapse_ab_initio, shift_ab_initio, spatial_metric_ab_initio);
  const auto& spacetime_unit_normal_one_form_ab_initio =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio);
  const auto spacetime_unit_normal_ab_initio =
      gr::spacetime_normal_vector<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio, shift_ab_initio);
  const auto& inverse_spatial_metric_ab_initio =
      determinant_and_inverse(spatial_metric_ab_initio).second;
  const auto inverse_spacetime_metric_ab_initio =
      gr::inverse_spacetime_metric<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio, shift_ab_initio, inverse_spatial_metric_ab_initio);
  const auto& det_spatial_metric_ab_initio =
      determinant_and_inverse(spatial_metric_ab_initio).first;
  Scalar<DataVector> sqrt_det_spatial_metric_ab_initio(
      sqrt(get(det_spatial_metric_ab_initio)));
  const auto phi_ab_initio = GeneralizedHarmonic::phi(
      lapse_ab_initio, d_lapse_ab_initio, shift_ab_initio, d_shift_ab_initio,
      spatial_metric_ab_initio, d_spatial_metric_ab_initio);
  const auto pi_ab_initio = GeneralizedHarmonic::pi(
      lapse_ab_initio, dt_lapse_ab_initio, shift_ab_initio, dt_shift_ab_initio,
      spatial_metric_ab_initio, dt_spatial_metric_ab_initio, phi_ab_initio);

  // commonly used terms
  const auto exp_fac_1 = 1. / 2.;
  const auto& log_fac_1 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::log_factor_metric_lapse<
          DataVector>(lapse_ab_initio, sqrt_det_spatial_metric_ab_initio,
                      exp_fac_1);

  // compute GR dependent terms
  // coeffs that enter gauge source function
  const auto mu_L1 =
      amp_coef_L1 * roll_on_L1 * get(weight) * pow(get(log_fac_1), exp_L1);

  // Calc \f$ \mu_1 = \mu_{L1} log(rootg/N) = R W log(rootg/N)^5\f$
  const auto mu1 = mu_L1 * get(log_fac_1);

  // Calc \f$ \partial_a [R W] \f$
  auto d4_RW_L1 = d4_weight;
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_RW_L1.get(a) *= roll_on_L1;
  }
  d4_RW_L1.get(0) += get(weight) * d0_roll_on_L1;

  // Calc derivs of \f$ \mu_1 \f$
  auto d4_mu1 =
      make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse), 0.);
  {
    const auto& d4_log_fac_mu1 =
        GeneralizedHarmonic::DampedHarmonicGauge_detail ::
            spacetime_deriv_of_power_log_factor_metric_lapse<
                SpatialDim, Frame::Inertial, DataVector>(
                lapse_ab_initio, shift_ab_initio,
                spacetime_unit_normal_ab_initio,
                inverse_spatial_metric_ab_initio,
                sqrt_det_spatial_metric_ab_initio, dt_spatial_metric_ab_initio,
                pi_ab_initio, phi_ab_initio, exp_fac_1, exp_L1 + 1);
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      // \f$ \partial_a \mu_1 \f$
      d4_mu1.get(a) =
          amp_coef_L1 * pow(get(log_fac_1), exp_L1 + 1) * d4_RW_L1.get(a) +
          amp_coef_L1 * roll_on_L1 * get(weight) * d4_log_fac_mu1.get(a);
    }
  }

  // Calc \f$ \partial_a N = {\partial_0 N, \partial_i N} \f$
  auto d4_N = make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
      get(lapse_ab_initio), 0.);
  d4_N.get(0) = get(dt_lapse_ab_initio);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_N.get(1 + i) = d_lapse_ab_initio.get(i);
  }

  // Calc \f$ \partial_a n_b = {-\partial_a N, 0, 0, 0} \f$
  auto d4_normal_one_form =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse_ab_initio), 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_normal_one_form.get(a, 0) -= d4_N.get(a);
  }

  // Calc \f$ \partial_a T2 \f$
  auto dT2 =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      dT2.get(a, b) +=
          mu1 * d4_normal_one_form.get(a, b) +
          d4_mu1.get(a) * spacetime_unit_normal_one_form_ab_initio.get(b);
    }
  }

  // Calc \f$ \partial_a H_b = dT2_{ab} \f$
  const auto& d4_gauge_h_expected = dT2;

  // Check that locally computed \f$\partial_a H_b\f$ matches returned one
  CHECK_ITERABLE_APPROX(d4_gauge_h_expected, d4_gauge_h);
}

// Test term 3/4 of spacetime deriv of gauge source function for Kerr-Schild
template <typename Solution>
void test_deriv_damped_harmonic_h_function_term_3_of_4_analytic_schwarzschild(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.01, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<SpatialDim>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<SpatialDim>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<SpatialDim>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<SpatialDim>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto spacetime_unit_normal =
      gr::spacetime_normal_vector<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift);
  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift, inverse_spatial_metric);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));
  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);

  // Initialize settings
  const double amp_coef_L2 = pdist(generator);
  const int exp_L2 = idist(generator);
  const double t_start_L2 = pdist(generator) * 0.1;
  const double sigma_t_L2 = pdist(generator) * 0.2;
  // weight function
  const double r_max = pdist(generator) * 0.7;

  // Initialize initial gauge function and its roll-off function
  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  const auto d4_gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<SpatialDim,
                                                             Frame::Inertial>>>(
      x, 0.);

  // compute d4H using function being tested
  typename db::item_type<GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<
      SpatialDim, Frame::Inertial>>
      d4_gauge_h{};
  GeneralizedHarmonic::spacetime_deriv_damped_harmonic_h<SpatialDim,
                                                         Frame::Inertial>(
      make_not_null(&d4_gauge_h), gauge_h_init, d4_gauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, t, x, 0., amp_coef_L2,
      0., 0, exp_L2, 0,  // exponents
      // roll on function parameters for lapse / shift terms
      -1., -1., -1., -1., t_start_L2, sigma_t_L2, -1., -1.,
      r_max /* weight function */);

  // Tempering functions
  const double roll_on_L2 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_L2, sigma_t_L2);
  const auto d0_roll_on_L2 = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      time_deriv_of_roll_on_function(t, t_start_L2, sigma_t_L2);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);
  auto d4_weight = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      spacetime_deriv_of_weight_function<SpatialDim, Frame::Inertial,
                                         DataVector>(x, r_max);

  // Evaluate analytic solution for Schwarzschild, ab initio.
  const double mass = solution.mass();
  const std::array<double, SpatialDim> center = solution.center();
  // 1) shift coords with BH at the center
  auto x_minus_center = x;
  // DataVector r_squared(get_size(x), 0.);
  auto r_squared = get<0>(x);
  r_squared -= get<0>(x);
  for (size_t i = 0; i < SpatialDim; ++i) {
    x_minus_center.get(i) -= gsl::at(center, i);
    r_squared += square(x_minus_center.get(i));
  }
  const DataVector r_ = sqrt(r_squared);
  // 2) compute scalar function that defines KerrSchild coords
  const DataVector H = mass * (1. / r_);
  const DataVector H2 = H * 2.;
  // 3) compute 3D null_one_form and its derivatives
  tnsr::i<DataVector, SpatialDim, Frame::Inertial> null_one_form{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    null_one_form.get(i) = x_minus_center.get(i) / r_;
  }
  auto null_vec = null_one_form;
  tnsr::ij<DataVector, SpatialDim, Frame::Inertial> deriv_null_one_form{};
  const DataVector denom = 1. / r_squared;
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      deriv_null_one_form.get(i, j) =
          -(null_one_form.get(i) * null_one_form.get(j)) / r_;
      if (i == j) {
        deriv_null_one_form.get(j, i) += (1. / r_);
      }
    }
  }
  // 4) compute deriv of H
  tnsr::i<DataVector, SpatialDim, Frame::Inertial> deriv_H{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    deriv_H.get(i) = -(H / r_squared) * x_minus_center.get(i);
  }
  // 5) compute lapse & its derivatives
  const Scalar<DataVector> lapse_ab_initio{sqrt(1. / (1. + H2))};
  const auto dt_lapse_ab_initio = make_with_value<Scalar<DataVector>>(x, 0.);
  auto d_lapse_ab_initio =
      make_with_value<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d_lapse_ab_initio.get(i) -= (pow<3>(get(lapse_ab_initio)) * deriv_H.get(i));
  }
  // 6) compute shift & its derivatives
  auto shift_ab_initio =
      make_with_value<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    shift_ab_initio.get(i) = (H2 / (1. + H2)) * null_vec.get(i);
  }
  const auto dt_shift_ab_initio =
      make_with_value<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  auto d_shift_ab_initio =
      make_with_value<tnsr::iJ<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      d_shift_ab_initio.get(i, j) =
          2. * pow<4>(get(lapse_ab_initio)) * null_vec.get(j) * deriv_H.get(i) +
          H2 * square(get(lapse_ab_initio)) * deriv_null_one_form.get(i, j);
    }
  }
  // 7) compute 3-metric & its derivatives
  auto spatial_metric_ab_initio =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      spatial_metric_ab_initio.get(i, j) =
          H2 * null_vec.get(i) * null_vec.get(j);
      if (i == j) {
        spatial_metric_ab_initio.get(i, i) += 1.;
      }
    }
  }
  auto dt_spatial_metric_ab_initio =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  auto d_spatial_metric_ab_initio =
      make_with_value<tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>>(x,
                                                                          0.);
  for (size_t k = 0; k < SpatialDim; ++k) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = i; j < SpatialDim; ++j) {
        d_spatial_metric_ab_initio.get(k, i, j) =
            2. * null_vec.get(i) * null_vec.get(j) * deriv_H.get(k) +
            H2 * (null_one_form.get(i) * deriv_null_one_form.get(k, j) +
                  null_one_form.get(j) * deriv_null_one_form.get(k, i));
      }
    }
  }

  const auto& spacetime_metric_ab_initio = gr::spacetime_metric(
      lapse_ab_initio, shift_ab_initio, spatial_metric_ab_initio);
  const auto& spacetime_unit_normal_one_form_ab_initio =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio);
  const auto spacetime_unit_normal_ab_initio =
      gr::spacetime_normal_vector<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio, shift_ab_initio);
  const auto& inverse_spatial_metric_ab_initio =
      determinant_and_inverse(spatial_metric_ab_initio).second;
  const auto inverse_spacetime_metric_ab_initio =
      gr::inverse_spacetime_metric<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio, shift_ab_initio, inverse_spatial_metric_ab_initio);
  const auto& det_spatial_metric_ab_initio =
      determinant_and_inverse(spatial_metric_ab_initio).first;
  Scalar<DataVector> sqrt_det_spatial_metric_ab_initio(
      sqrt(get(det_spatial_metric_ab_initio)));
  const auto phi_ab_initio = GeneralizedHarmonic::phi(
      lapse_ab_initio, d_lapse_ab_initio, shift_ab_initio, d_shift_ab_initio,
      spatial_metric_ab_initio, d_spatial_metric_ab_initio);
  const auto pi_ab_initio = GeneralizedHarmonic::pi(
      lapse_ab_initio, dt_lapse_ab_initio, shift_ab_initio, dt_shift_ab_initio,
      spatial_metric_ab_initio, dt_spatial_metric_ab_initio, phi_ab_initio);

  // commonly used terms
  const auto exp_fac_2 = 0.;
  const auto& log_fac_2 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::log_factor_metric_lapse<
          DataVector>(lapse_ab_initio, sqrt_det_spatial_metric_ab_initio,
                      exp_fac_2);

  // coeffs that enter gauge source function
  const auto mu_L2 =
      amp_coef_L2 * roll_on_L2 * get(weight) * pow(get(log_fac_2), exp_L2);

  // Calc \f$ \mu_2 = \mu_{L2} log(1/N) = R W log(1/N)^5\f$
  const auto mu2 = mu_L2 * get(log_fac_2);

  // Calc \f$ \partial_a [R W] \f$
  auto d4_RW_L2 = d4_weight;
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_RW_L2.get(a) *= roll_on_L2;
  }
  d4_RW_L2.get(0) += get(weight) * d0_roll_on_L2;

  // Calc derivs of \f$ \mu_2 \f$
  auto d4_mu2 =
      make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse_ab_initio), 0.);
  {
    const auto& d4_log_fac_mu2 =
        GeneralizedHarmonic::DampedHarmonicGauge_detail ::
            spacetime_deriv_of_power_log_factor_metric_lapse<
                SpatialDim, Frame::Inertial, DataVector>(
                lapse_ab_initio, shift_ab_initio,
                spacetime_unit_normal_ab_initio,
                inverse_spatial_metric_ab_initio,
                sqrt_det_spatial_metric_ab_initio, dt_spatial_metric_ab_initio,
                pi_ab_initio, phi_ab_initio, exp_fac_2, exp_L2 + 1);
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      // \f$ \partial_a \mu_2 \f$
      d4_mu2.get(a) =
          amp_coef_L2 * pow(get(log_fac_2), exp_L2 + 1) * d4_RW_L2.get(a) +
          amp_coef_L2 * roll_on_L2 * get(weight) * d4_log_fac_mu2.get(a);
    }
  }

  // Calc \f$ \partial_a N = {\partial_0 N, \partial_i N} \f$
  auto d4_N = make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
      get(lapse_ab_initio), 0.);
  d4_N.get(0) = get(dt_lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_N.get(1 + i) = d_lapse_ab_initio.get(i);
  }

  // Calc \f$ \partial_a n_b = {-\partial_a N, 0, 0, 0} \f$
  auto d4_normal_one_form =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse_ab_initio), 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_normal_one_form.get(a, 0) -= d4_N.get(a);
  }

  // Calc \f$ \partial_a T2 \f$
  auto dT2 =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      dT2.get(a, b) +=
          mu2 * d4_normal_one_form.get(a, b) +
          d4_mu2.get(a) * spacetime_unit_normal_one_form_ab_initio.get(b);
    }
  }

  // Calc \f$ \partial_a H_b = dT2_{ab} \f$
  const auto& d4_gauge_h_expected = dT2;

  // Check that locally computed \f$\partial_a H_b\f$ matches returned one
  CHECK_ITERABLE_APPROX(d4_gauge_h_expected, d4_gauge_h);
}

// Test term 4/4 of spacetime deriv of gauge source function for Kerr-Schild
template <typename Solution>
void test_deriv_damped_harmonic_h_function_term_4_of_4_analytic_schwarzschild(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.01, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<SpatialDim>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<SpatialDim>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<SpatialDim>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<SpatialDim>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto spacetime_unit_normal =
      gr::spacetime_normal_vector<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift);
  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift, inverse_spatial_metric);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));
  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);
  const auto d4_spacetime_metric = gr::derivatives_of_spacetime_metric(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, spatial_metric,
      dt_spatial_metric, d_spatial_metric);

  // Initialize settings
  const double amp_coef_S = pdist(generator);
  const int exp_S = idist(generator);
  const double t_start_S = pdist(generator) * 0.1;
  const double sigma_t_S = pdist(generator) * 0.2;
  const double r_max = pdist(generator) * 0.7;

  // Initialize initial gauge function and its roll-off function
  const auto gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      x, 0.);
  const auto d4_gauge_h_init = make_with_value<typename db::item_type<
      GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<SpatialDim,
                                                             Frame::Inertial>>>(
      x, 0.);

  // compute d4H using function being tested
  typename db::item_type<GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<
      SpatialDim, Frame::Inertial>>
      d4_gauge_h{};
  GeneralizedHarmonic::spacetime_deriv_damped_harmonic_h<SpatialDim,
                                                         Frame::Inertial>(
      make_not_null(&d4_gauge_h), gauge_h_init, d4_gauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, t, x, 0., 0.,
      amp_coef_S, 0, 0, exp_S,  // exponents
      // roll on function parameters for lapse / shift terms
      -1., -1., -1., -1., -1., -1., t_start_S, sigma_t_S,
      r_max /* weight function */);

  // Tempering functions
  const double roll_on_S =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::roll_on_function(
          t, t_start_S, sigma_t_S);
  const auto d0_roll_on_S = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      time_deriv_of_roll_on_function(t, t_start_S, sigma_t_S);
  const auto& weight =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::weight_function<
          SpatialDim, Frame::Inertial, DataVector>(x, r_max);
  auto d4_weight = GeneralizedHarmonic::DampedHarmonicGauge_detail::
      spacetime_deriv_of_weight_function<SpatialDim, Frame::Inertial,
                                         DataVector>(x, r_max);

  // Evaluate analytic solution for Schwarzschild, ab initio.
  const double mass = solution.mass();
  const std::array<double, SpatialDim> center = solution.center();
  // 1) shift coords with BH at the center
  auto x_minus_center = x;
  // DataVector r_squared(get_size(x), 0.);
  auto r_squared = get<0>(x);
  r_squared -= get<0>(x);
  for (size_t i = 0; i < SpatialDim; ++i) {
    x_minus_center.get(i) -= gsl::at(center, i);
    r_squared += square(x_minus_center.get(i));
  }
  const DataVector r_ = sqrt(r_squared);
  // 2) compute scalar function that defines KerrSchild coords
  const DataVector H = mass * (1. / r_);
  const DataVector H2 = H * 2.;
  // 3) compute 3D null_one_form and its derivatives
  tnsr::i<DataVector, SpatialDim, Frame::Inertial> null_one_form{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    null_one_form.get(i) = x_minus_center.get(i) / r_;
  }
  auto null_vec = null_one_form;
  tnsr::ij<DataVector, SpatialDim, Frame::Inertial> deriv_null_one_form{};
  const DataVector denom = 1. / r_squared;
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      deriv_null_one_form.get(i, j) =
          -(null_one_form.get(i) * null_one_form.get(j)) / r_;
      if (i == j) {
        deriv_null_one_form.get(j, i) += (1. / r_);
      }
    }
  }
  // 4) compute deriv of H
  tnsr::i<DataVector, SpatialDim, Frame::Inertial> deriv_H{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    deriv_H.get(i) = -(H / r_squared) * x_minus_center.get(i);
  }
  // 5) compute lapse & its derivatives
  const Scalar<DataVector> lapse_ab_initio{sqrt(1. / (1. + H2))};
  const auto dt_lapse_ab_initio = make_with_value<Scalar<DataVector>>(x, 0.);
  auto d_lapse_ab_initio =
      make_with_value<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d_lapse_ab_initio.get(i) -= (pow<3>(get(lapse_ab_initio)) * deriv_H.get(i));
  }
  // 6) compute shift & its derivatives
  auto shift_ab_initio =
      make_with_value<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    shift_ab_initio.get(i) = (H2 / (1. + H2)) * null_vec.get(i);
  }
  const auto dt_shift_ab_initio =
      make_with_value<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  auto d_shift_ab_initio =
      make_with_value<tnsr::iJ<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      d_shift_ab_initio.get(i, j) =
          2. * pow<4>(get(lapse_ab_initio)) * null_vec.get(j) * deriv_H.get(i) +
          H2 * square(get(lapse_ab_initio)) * deriv_null_one_form.get(i, j);
    }
  }
  // 7) compute 3-metric & its derivatives
  auto spatial_metric_ab_initio =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      spatial_metric_ab_initio.get(i, j) =
          H2 * null_vec.get(i) * null_vec.get(j);
      if (i == j) {
        spatial_metric_ab_initio.get(i, i) += 1.;
      }
    }
  }
  auto dt_spatial_metric_ab_initio =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  auto d_spatial_metric_ab_initio =
      make_with_value<tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>>(x,
                                                                          0.);
  for (size_t k = 0; k < SpatialDim; ++k) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = i; j < SpatialDim; ++j) {
        d_spatial_metric_ab_initio.get(k, i, j) =
            2. * null_vec.get(i) * null_vec.get(j) * deriv_H.get(k) +
            H2 * (null_one_form.get(i) * deriv_null_one_form.get(k, j) +
                  null_one_form.get(j) * deriv_null_one_form.get(k, i));
      }
    }
  }

  const auto& spacetime_metric_ab_initio = gr::spacetime_metric(
      lapse_ab_initio, shift_ab_initio, spatial_metric_ab_initio);
  const auto& spacetime_unit_normal_one_form_ab_initio =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio);
  const auto spacetime_unit_normal_ab_initio =
      gr::spacetime_normal_vector<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio, shift_ab_initio);
  const auto& inverse_spatial_metric_ab_initio =
      determinant_and_inverse(spatial_metric_ab_initio).second;
  const auto inverse_spacetime_metric_ab_initio =
      gr::inverse_spacetime_metric<SpatialDim, Frame::Inertial, DataVector>(
          lapse_ab_initio, shift_ab_initio, inverse_spatial_metric_ab_initio);
  const auto& det_spatial_metric_ab_initio =
      determinant_and_inverse(spatial_metric_ab_initio).first;
  Scalar<DataVector> sqrt_det_spatial_metric_ab_initio(
      sqrt(get(det_spatial_metric_ab_initio)));
  const auto phi_ab_initio = GeneralizedHarmonic::phi(
      lapse_ab_initio, d_lapse_ab_initio, shift_ab_initio, d_shift_ab_initio,
      spatial_metric_ab_initio, d_spatial_metric_ab_initio);
  const auto pi_ab_initio = GeneralizedHarmonic::pi(
      lapse_ab_initio, dt_lapse_ab_initio, shift_ab_initio, dt_shift_ab_initio,
      spatial_metric_ab_initio, dt_spatial_metric_ab_initio, phi_ab_initio);

  // commonly used terms
  const auto exp_fac_1 = 1. / 2.;
  const auto& one_over_lapse = 1. / get(lapse_ab_initio);
  const auto& log_fac_1 =
      GeneralizedHarmonic::DampedHarmonicGauge_detail::log_factor_metric_lapse<
          DataVector>(lapse_ab_initio, sqrt_det_spatial_metric_ab_initio,
                      exp_fac_1);

  // coeffs that enter gauge source function
  const auto mu_S =
      amp_coef_S * roll_on_S * get(weight) * pow(get(log_fac_1), exp_S);
  const auto mu_S_over_N = mu_S * one_over_lapse;

  // Calc \f$ \partial_a [R W] \f$
  auto d4_RW_S = d4_weight;
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    d4_RW_S.get(a) *= roll_on_S;
  }
  d4_RW_S.get(0) += get(weight) * d0_roll_on_S;

  // Calc derivs of \f$ \mu_S \f$
  auto d4_mu_S =
      make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse_ab_initio), 0.);
  {
    const auto& d4_log_fac_muS =
        GeneralizedHarmonic::DampedHarmonicGauge_detail ::
            spacetime_deriv_of_power_log_factor_metric_lapse<
                SpatialDim, Frame::Inertial, DataVector>(
                lapse_ab_initio, shift_ab_initio,
                spacetime_unit_normal_ab_initio,
                inverse_spatial_metric_ab_initio,
                sqrt_det_spatial_metric_ab_initio, dt_spatial_metric_ab_initio,
                pi_ab_initio, phi_ab_initio, exp_fac_1, exp_S);
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      // \f$ \partial_a \mu_{L1} \f$
      d4_mu_S.get(a) =
          amp_coef_S * d4_RW_S.get(a) * pow(get(log_fac_1), exp_S) +
          amp_coef_S * roll_on_S * get(weight) * d4_log_fac_muS.get(a);
    }
  }

  // Calc \f$ \partial_a N = {\partial_0 N, \partial_i N} \f$
  auto d4_N = make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
      get(lapse_ab_initio), 0.);
  d4_N.get(0) = get(dt_lapse_ab_initio);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_N.get(1 + i) = d_lapse_ab_initio.get(i);
  }

  // Calc \f$ \partial_a N^i = {\partial_0 N^i, \partial_j N^i} \f$
  auto d4_shift =
      make_with_value<tnsr::aB<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse_ab_initio), 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_shift.get(0, 1 + i) = dt_shift_ab_initio.get(i);
    for (size_t j = 0; j < SpatialDim; ++j) {
      d4_shift.get(1 + j, 1 + i) = d_shift_ab_initio.get(j, i);
    }
  }

  // \f[ \partial_a (\mu_S/N) = (1/N) \partial_a \mu_{S}
  //         - (\mu_{S}/N^2) \partial_a N
  // \f]
  auto d4_muS_over_N =
      make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
          get(lapse_ab_initio), 0.);
  {
    auto prefac = -mu_S * one_over_lapse * one_over_lapse;
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_muS_over_N.get(a) +=
          one_over_lapse * d4_mu_S.get(a) + prefac * d4_N.get(a);
    }
  }

  // Calc \f$ \partial_a T3 \f$ (note minus sign)
  auto dT3 =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        dT3.get(a, b) -= d4_muS_over_N.get(a) *
                         spacetime_metric_ab_initio.get(b, i + 1) *
                         shift.get(i);
        dT3.get(a, b) -= mu_S_over_N * d4_spacetime_metric.get(a, b, i + 1) *
                         shift_ab_initio.get(i);
        dT3.get(a, b) -= mu_S_over_N *
                         spacetime_metric_ab_initio.get(b, i + 1) *
                         d4_shift.get(a, i + 1);
      }
    }
  }

  // Calc \f$ \partial_a H_b = dT3_{ab} \f$
  const auto& d4_gauge_h_expected = dT3;

  // Check that locally computed \f$\partial_a H_b\f$ matches returned one
  CHECK_ITERABLE_APPROX(d4_gauge_h_expected, d4_gauge_h);
}

//
//  Test ComputeTags
//
void test_damped_harmonic_compute_tags(const size_t grid_size_each_dimension,
                                       const std::array<double, 3>& lower_bound,
                                       const std::array<double, 3>& upper_bound,
                                       std::mt19937 generator) noexcept {
  // Initialize random number generation
  std::uniform_real_distribution<> pdist(0.1, 1.);
  std::uniform_int_distribution<> idist(2, 7);

  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = pdist(generator) * 0.4;
  const Slab slab(0, t);
  const Rational frac(1);
  const Time current_time(slab, frac);
  const TimeId current_time_id(true, 0, current_time);

  // Randomized 3 + 1 quantities
  // Note: Ranges from which random numbers are drawn to populate 3+1 tensors
  //       below were chosen to not throw FPEs (by trial & error).
  std::uniform_real_distribution<> dist(0.9, 1.);
  const DataVector& used_for_size = get<0>(x);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto dt_lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto d_lapse =
      make_with_random_values<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto dt_shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto d_shift = make_with_random_values<
      tnsr::iJ<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  get<0, 0>(spatial_metric) = get<1, 1>(spatial_metric) =
      get<2, 2>(spatial_metric) = 1.;
  std::uniform_real_distribution<> dist2(0., 0.12);
  const auto dspatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      spatial_metric.get(i, j) += dspatial_metric.get(i, j);
    }
  }
  const auto dt_spatial_metric = make_with_random_values<
      tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);
  const auto d_spatial_metric = make_with_random_values<
      tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>>(
      make_not_null(&generator), make_not_null(&dist2), used_for_size);

  // Get ingredients
  const auto& spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto& spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto spacetime_unit_normal =
      gr::spacetime_normal_vector<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift);
  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift, inverse_spatial_metric);
  const auto& det_spatial_metric =
      determinant_and_inverse(spatial_metric).first;
  Scalar<DataVector> sqrt_det_spatial_metric(sqrt(get(det_spatial_metric)));
  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);
  const auto d4_spacetime_metric = gr::derivatives_of_spacetime_metric(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, spatial_metric,
      dt_spatial_metric, d_spatial_metric);

  // Initialize settings
  const double t_start_S = pdist(generator) * 0.1;
  const double sigma_t_S = pdist(generator) * 0.2;
  const double r_max = pdist(generator) * 0.7;

  // initial H_a and D4(H_a)
  const auto gauge_h_init = make_with_random_values<typename db::item_type<
      GeneralizedHarmonic::Tags::InitialGaugeH<SpatialDim, Frame::Inertial>>>(
      make_not_null(&generator), make_not_null(&pdist), x);
  const auto d4_gauge_h_init = make_with_random_values<typename db::item_type<
      GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<SpatialDim,
                                                             Frame::Inertial>>>(
      make_not_null(&generator), make_not_null(&pdist), x);
  // local H_a
  const auto gauge_h_expected = wrap_DampedHarmonicHCompute(
      gauge_h_init, lapse, shift, sqrt_det_spatial_metric, spacetime_metric, t,
      t_start_S, sigma_t_S, x, r_max);
  // local D4(H_a)
  const auto d4_gauge_h_expected = wrap_SpacetimeDerivDampedHarmonicHCompute(
      gauge_h_init, d4_gauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, t, t_start_S,
      sigma_t_S, x, r_max);

  //
  // Check that compute items work correctly in the DataBox
  //
  // First, check that the names are correct
  CHECK(
      GeneralizedHarmonic::DampedHarmonicHCompute<3, Frame::Inertial>::name() ==
      "GaugeH");
  CHECK(GeneralizedHarmonic::SpacetimeDerivDampedHarmonicHCompute<
            3, Frame::Inertial>::name() == "SpacetimeDerivGaugeH");

  const auto box = db::create<
      db::AddSimpleTags<
          GeneralizedHarmonic::Tags::InitialGaugeH<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<
              3, Frame::Inertial>,
          gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<3, Frame::Inertial, DataVector>,
          gr::Tags::SpacetimeNormalOneForm<3, Frame::Inertial, DataVector>,
          gr::Tags::SqrtDetSpatialMetric<DataVector>,
          gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
          gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>,
          GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>, ::Tags::TimeId,
          ::Tags::Coordinates<3, Frame::Inertial>,
          GeneralizedHarmonic::OptionTags::GaugeHRollOnStartTime,
          GeneralizedHarmonic::OptionTags::GaugeHRollOnTimeWindow,
          GeneralizedHarmonic::OptionTags::GaugeHSpatialWeightDecayWidth<
              Frame::Inertial>>,
      db::AddComputeTags<
          ::Tags::Time,
          GeneralizedHarmonic::DampedHarmonicHCompute<3, Frame::Inertial>,
          GeneralizedHarmonic::SpacetimeDerivDampedHarmonicHCompute<
              3, Frame::Inertial>>>(
      gauge_h_init, d4_gauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, current_time_id, x,
      t_start_S, sigma_t_S, r_max);

  // Verify that locally computed H_a matches the same obtained through its
  // ComputeTag from databox
  CHECK(db::get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(box) == shift);
  CHECK(db::get<::Tags::Time>(box) == current_time);
  CHECK(db::get<GeneralizedHarmonic::OptionTags::GaugeHRollOnStartTime>(box) ==
        t_start_S);
  CHECK(db::get<GeneralizedHarmonic::OptionTags::GaugeHRollOnTimeWindow>(box) ==
        sigma_t_S);
  CHECK(db::get<GeneralizedHarmonic::OptionTags::GaugeHSpatialWeightDecayWidth<
            Frame::Inertial>>(box) == r_max);
  CHECK(db::get<GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>>(box) ==
        gauge_h_expected);
  CHECK(
      db::get<
          GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<3, Frame::Inertial>>(
          box) == d4_gauge_h_expected);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.DampedHarmonic.details",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  const DataVector used_for_size(4);

  test_options<Frame::Inertial>();

  test_detail_functions<1, Frame::Inertial>(used_for_size);
  test_detail_functions<2, Frame::Inertial>(used_for_size);
  test_detail_functions<3, Frame::Inertial>(used_for_size);

  test_detail_functions<1, Frame::Inertial>(1.);
  test_detail_functions<2, Frame::Inertial>(1.);
  test_detail_functions<3, Frame::Inertial>(1.);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.DampedHarmonic.H",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  const DataVector used_for_size(4);

  // Compare with Python implementation
  test_damped_harmonic_h_function<1, Frame::Inertial>(used_for_size);
  test_damped_harmonic_h_function<2, Frame::Inertial>(used_for_size);
  test_damped_harmonic_h_function<3, Frame::Inertial>(used_for_size);

  // Piece-wise tests with random tensors
  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};

  MAKE_GENERATOR(generator);
  test_damped_harmonic_h_function_term_1_of_4(grid_size, lower_bound,
                                              upper_bound, generator);
  test_damped_harmonic_h_function_term_2_of_4(grid_size, lower_bound,
                                              upper_bound, generator);
  test_damped_harmonic_h_function_term_3_of_4(grid_size, lower_bound,
                                              upper_bound, generator);
  test_damped_harmonic_h_function_term_4_of_4(grid_size, lower_bound,
                                              upper_bound, generator);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.DampedHarmonic.D4H",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  const DataVector used_for_size(4);

  // Compare with Python implementation
  test_deriv_damped_harmonic_h_function<1, Frame::Inertial>(used_for_size);
  test_deriv_damped_harmonic_h_function<2, Frame::Inertial>(used_for_size);
  test_deriv_damped_harmonic_h_function<3, Frame::Inertial>(used_for_size);

  // Piece-wise tests with random tensors
  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};

  MAKE_GENERATOR(generator);
  test_deriv_damped_harmonic_h_function_term_1_of_4(grid_size, lower_bound,
                                                    upper_bound, generator);
  test_deriv_damped_harmonic_h_function_term_2_of_4(grid_size, lower_bound,
                                                    upper_bound, generator);
  test_deriv_damped_harmonic_h_function_term_3_of_4(grid_size, lower_bound,
                                                    upper_bound, generator);
  test_deriv_damped_harmonic_h_function_term_4_of_4(grid_size, lower_bound,
                                                    upper_bound, generator);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.DampedHarmonic.Analytic",
    "[Unit][Evolution]") {
  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};
  MAKE_GENERATOR(generator);

  // Analytic spacetime tests
  const double mass = 2.;
  const std::array<double, 3> spin{{0., 0., 0.}};
  const std::array<double, 3> center{{0.2, -0.3, 0.5}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  test_damped_harmonic_h_function_term_2_of_4_analytic_schwarzschild(
      solution, grid_size, lower_bound, upper_bound, generator);
  test_damped_harmonic_h_function_term_3_of_4_analytic_schwarzschild(
      solution, grid_size, lower_bound, upper_bound, generator);
  test_damped_harmonic_h_function_term_4_of_4_analytic_schwarzschild(
      solution, grid_size, lower_bound, upper_bound, generator);

  test_deriv_damped_harmonic_h_function_term_2_of_4_analytic_schwarzschild(
      solution, grid_size, lower_bound, upper_bound, generator);
  test_deriv_damped_harmonic_h_function_term_3_of_4_analytic_schwarzschild(
      solution, grid_size, lower_bound, upper_bound, generator);
  test_deriv_damped_harmonic_h_function_term_4_of_4_analytic_schwarzschild(
      solution, grid_size, lower_bound, upper_bound, generator);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.DampedHarmonic.CompTags",
    "[Unit][Evolution]") {
  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};
  MAKE_GENERATOR(generator);

  // ComputeTag tests
  test_damped_harmonic_compute_tags(grid_size, lower_bound, upper_bound,
                                    generator);
}
