// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedWaveHelpers.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
void test_detail_functions(const DataType& used_for_size) {
  // weight_function
  pypp::check_with_random_values<2>(
      &::gh::gauges::DampedHarmonicGauge_detail::spatial_weight_function<
          DataType, SpatialDim, Frame>,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedWaveHelpers",
      {"spatial_weight_function"},
      {{{-10., 10.}, {std::numeric_limits<double>::denorm_min(), 10.}}},
      used_for_size);
  // spacetime_deriv_of_spatial_weight_function
  pypp::check_with_random_values<3>(
      &::gh::gauges::DampedHarmonicGauge_detail::
          spacetime_deriv_of_spatial_weight_function<DataType, SpatialDim,
                                                     Frame>,
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
          &::gh::gauges::DampedHarmonicGauge_detail::log_factor_metric_lapse<
              DataType>),
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedWaveHelpers",
      "log_fac", {{{std::numeric_limits<double>::denorm_min(), 10.}}},
      used_for_size);
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
