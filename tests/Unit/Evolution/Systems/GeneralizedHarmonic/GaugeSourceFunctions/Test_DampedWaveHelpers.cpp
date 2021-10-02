// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedWaveHelpers.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"

namespace {
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
    const double d_exponent) {
  const auto exponent = static_cast<int>(d_exponent);
  tnsr::a<DataType, SpatialDim, Frame> d4_powlogfac{};
  GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail::
      spacetime_deriv_of_power_log_factor_metric_lapse(
          make_not_null(&d4_powlogfac), lapse, shift, spacetime_unit_normal,
          inverse_spatial_metric, sqrt_det_spatial_metric, dt_spatial_metric,
          pi, phi, g_exponent, exponent);
  return d4_powlogfac;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void test_detail_functions(const DataType& used_for_size) {
  // weight_function
  pypp::check_with_random_values<2>(
      &::GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail::
          spatial_weight_function<SpatialDim, Frame, DataType>,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedWaveHelpers",
      {"spatial_weight_function"},
      {{{-10., 10.}, {std::numeric_limits<double>::denorm_min(), 10.}}},
      used_for_size);
  // spacetime_deriv_of_spatial_weight_function
  pypp::check_with_random_values<3>(
      &::GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail::
          spacetime_deriv_of_spatial_weight_function<SpatialDim, Frame,
                                                     DataType>,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedWaveHelpers",
      {"spacetime_deriv_spatial_weight_function"},
      {{{-10., 10.},
        {std::numeric_limits<double>::denorm_min(), 10.},
        {std::numeric_limits<double>::denorm_min(), 10.}}},
      used_for_size);
  // log_factor_metric_lapse
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(const Scalar<DataType>&,
                                       const Scalar<DataType>&, const double)>(
          &::GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail::
              log_factor_metric_lapse<DataType>),
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedWaveHelpers",
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
          &::GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail::
              spacetime_deriv_of_log_factor_metric_lapse<SpatialDim, Frame,
                                                         DataType>),
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedWaveHelpers",
      "spacetime_deriv_log_fac",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size,
      1.0e-11);
  // spacetime_deriv_of_power_log_factor_metric_lapse
  pypp::check_with_random_values<1>(
      &wrap_spacetime_deriv_of_power_log_factor_metric_lapse<SpatialDim, Frame,
                                                             DataType>,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedWaveHelpers",
      "spacetime_deriv_pow_log_fac",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size,
      1.e-11);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Gh.Gauge.DampedWaveHelpers",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  const DataVector used_for_size(4);

  test_detail_functions<1, Frame::Inertial>(used_for_size);
  test_detail_functions<2, Frame::Inertial>(used_for_size);
  test_detail_functions<3, Frame::Inertial>(used_for_size);

  test_detail_functions<1, Frame::Inertial>(1.);
  test_detail_functions<2, Frame::Inertial>(1.);
  test_detail_functions<3, Frame::Inertial>(1.);
}
