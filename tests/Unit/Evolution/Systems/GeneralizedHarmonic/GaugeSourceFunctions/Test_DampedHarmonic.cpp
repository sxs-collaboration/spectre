// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

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
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedWaveHelpers.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail {
// The return-by-value implementations of spatial_weight_function and
// spacetime_deriv_of_spatial_weight_function are intentionally only available
// in the test because while convenient the additional allocations are bad for
// performance. By not having them available in the production code we avoid
// possible accidental usage.
template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> spatial_weight_function(
    const tnsr::I<DataType, SpatialDim, Frame>& coords,
    const double sigma_r) noexcept {
  Scalar<DataType> spatial_weight{};
  spatial_weight_function(make_not_null(&spatial_weight), coords, sigma_r);
  return spatial_weight;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_spatial_weight_function(
    const tnsr::I<DataType, SpatialDim, Frame>& coords,
    const double sigma_r) noexcept {
  tnsr::a<DataType, SpatialDim, Frame> d4_weight{};
  spacetime_deriv_of_spatial_weight_function(
      make_not_null(&d4_weight), coords, sigma_r,
      spatial_weight_function(coords, sigma_r));
  return d4_weight;
}

double roll_on_function(double time, double t_start, double sigma_t) noexcept;

double time_deriv_of_roll_on_function(double time, double t_start,
                                      double sigma_t) noexcept;
}  // namespace GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
void test_rollon_function(const DataType& used_for_size) noexcept {
  INFO("Test rollon function");
  // roll_on_function
  pypp::check_with_random_values<1>(
      &::GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail::
          roll_on_function,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "roll_on_function", {{{std::numeric_limits<double>::denorm_min(), 1.}}},
      used_for_size);
  // time_deriv_of_roll_on_function
  pypp::check_with_random_values<1>(
      &::GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail::
          time_deriv_of_roll_on_function,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "time_deriv_roll_on_function",
      {{{std::numeric_limits<double>::denorm_min(), 1.}}}, used_for_size);
}

//  Tests of the damped harmonic gauge source function and its spacetime
//  derivative. We need a wrapper because pypp does not currently support
//  integer types.
template <size_t SpatialDim, typename Frame>
void wrap_damped_harmonic_rollon(
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
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, double time,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double amp_coef_L1, const double amp_coef_L2, const double amp_coef_S,
    const double rollon_start_time, const double rollon_width,
    const double sigma_r) noexcept {
  GeneralizedHarmonic::gauges::damped_harmonic_rollon(
      gauge_h, d4_gauge_h, gauge_h_init, dgauge_h_init, lapse, shift,
      spacetime_unit_normal_one_form, sqrt_det_spatial_metric,
      inverse_spatial_metric, spacetime_metric, pi, phi, time, coords,
      amp_coef_L1, amp_coef_L2, amp_coef_S, 4, 4, 4, rollon_start_time,
      rollon_width, sigma_r);
}

template <size_t SpatialDim, typename Frame>
void wrap_damped_harmonic(
    const gsl::not_null<tnsr::a<DataVector, SpatialDim, Frame>*> gauge_h,
    const gsl::not_null<tnsr::ab<DataVector, SpatialDim, Frame>*> d4_gauge_h,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const tnsr::a<DataVector, SpatialDim, Frame>&
        spacetime_unit_normal_one_form,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double amp_coef_L1, const double amp_coef_L2, const double amp_coef_S,
    const double sigma_r) noexcept {
  GeneralizedHarmonic::gauges::damped_harmonic(
      gauge_h, d4_gauge_h, lapse, shift, spacetime_unit_normal_one_form,
      sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric, pi,
      phi, coords, amp_coef_L1, amp_coef_L2, amp_coef_S, 4, 4, 4, sigma_r);
}

// Compare with Python implementation
template <size_t SpatialDim, typename Frame>
void test_with_python(const DataVector& used_for_size) noexcept {
  INFO("Test with python");
  CAPTURE(SpatialDim);
  CAPTURE(Frame{});
  pypp::check_with_random_values<1>(
      &wrap_damped_harmonic_rollon<SpatialDim, Frame>,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      {"damped_harmonic_gauge_source_function_rollon",
       "spacetime_deriv_damped_harmonic_gauge_source_function_rollon"},
      {{{0.1, 1.}}}, used_for_size);

  pypp::check_with_random_values<1>(
      &wrap_damped_harmonic<SpatialDim, Frame>,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      {"damped_harmonic_gauge_source_function",
       "spacetime_deriv_damped_harmonic_gauge_source_function"},
      {{{0.1, 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.DampedHarmonic",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  const DataVector used_for_size(5);

  test_rollon_function<1, Frame::Inertial>(used_for_size);
  test_rollon_function<2, Frame::Inertial>(used_for_size);
  test_rollon_function<3, Frame::Inertial>(used_for_size);

  test_rollon_function<1, Frame::Inertial>(1.);
  test_rollon_function<2, Frame::Inertial>(1.);
  test_rollon_function<3, Frame::Inertial>(1.);

  // Compare with Python implementation
  test_with_python<1, Frame::Inertial>(used_for_size);
  test_with_python<2, Frame::Inertial>(used_for_size);
  test_with_python<3, Frame::Inertial>(used_for_size);
}
