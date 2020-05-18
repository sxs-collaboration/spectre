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
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
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
template <typename Frame>
void test_options() noexcept {
  Options<tmpl::list<
      GeneralizedHarmonic::OptionTags::GaugeHRollOnStart,
      GeneralizedHarmonic::OptionTags::GaugeHRollOnWindow,
      GeneralizedHarmonic::OptionTags::GaugeHSpatialDecayWidth<Frame>>>
      opts("");
  opts.parse(
      "EvolutionSystem:\n"
      "  GeneralizedHarmonic:\n"
      "    Gauge:\n"
      "      RollOnStartTime : 0.\n"
      "      RollOnTimeWindow : 100.\n"
      "      SpatialDecayWidth : 50.\n");
  CHECK(
      opts.template get<GeneralizedHarmonic::OptionTags::GaugeHRollOnStart>() ==
      0.);
  CHECK(opts.template get<
            GeneralizedHarmonic::OptionTags::GaugeHRollOnWindow>() == 100.);
  CHECK(
      opts.template get<
          GeneralizedHarmonic::OptionTags::GaugeHSpatialDecayWidth<Frame>>() ==
      50.);
}

template <size_t SpatialDim, typename Frame, typename DataType>
void test_rollon_function(const DataType& used_for_size) noexcept {
  // roll_on_function
  pypp::check_with_random_values<1>(
      &::GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail::
          roll_on_function,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "roll_on_function", {{{std::numeric_limits<double>::denorm_min(), 10.}}},
      used_for_size);
  // time_deriv_of_roll_on_function
  pypp::check_with_random_values<1>(
      &::GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail::
          time_deriv_of_roll_on_function,
      "Evolution.Systems.GeneralizedHarmonic.GaugeSourceFunctions."
      "DampedHarmonic",
      "time_deriv_roll_on_function",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size);
}

// Wrap `DampedHarmonicHCompute::function` here to make its time
// argument a double, allowing for `pypp::check_with_random_values` to work.
template <size_t SpatialDim, typename Frame>
tnsr::a<DataVector, SpatialDim, Frame> wrap_DampedHarmonicHCompute(
    const tnsr::a<DataVector, SpatialDim, Frame>& gauge_h_init,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::aa<DataVector, SpatialDim, Frame>& spacetime_metric,
    const double t, const double t_start, const double sigma_t,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double sigma_r) noexcept {
  tnsr::a<DataVector, SpatialDim, Frame> gauge_h{};
  tnsr::ab<DataVector, SpatialDim, Frame> d4_gauge_h{};
  const auto dgauge_h_init =
      make_with_value<tnsr::ab<DataVector, SpatialDim, Frame>>(lapse, 0.);
  const auto spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame, DataVector>(lapse);
  const auto pi =
      make_with_value<tnsr::aa<DataVector, SpatialDim, Frame>>(lapse, 0.);
  const auto phi =
      make_with_value<tnsr::iaa<DataVector, SpatialDim, Frame>>(lapse, 0.);
  auto inverse_spatial_metric =
      make_with_value<tnsr::II<DataVector, SpatialDim, Frame>>(lapse, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    inverse_spatial_metric.get(i, i) = 1.;
  }
  GeneralizedHarmonic::gauges::damped_harmonic_rollon(
      make_not_null(&gauge_h), make_not_null(&d4_gauge_h), gauge_h_init,
      dgauge_h_init, lapse, shift, spacetime_unit_normal_one_form,
      sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric, pi,
      phi, t, coords, 1., 1.,
      1.,       // amp_coef_{L1, L2, S}
      4, 4, 4,  // exp_{L1, L2, S}
      t_start, sigma_t, sigma_r);
  return gauge_h;
}

// Wrap `SpacetimeDerivDampedHarmonicHCompute::function` here to make its time
// argument a double, allowing for `pypp::check_with_random_values` to work.
template <size_t SpatialDim, typename Frame>
tnsr::ab<DataVector, SpatialDim, Frame>
wrap_SpacetimeDerivDampedHarmonicHCompute(
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
    const tnsr::iaa<DataVector, SpatialDim, Frame>& phi, const double t,
    const double t_start, const double sigma_t,
    const tnsr::I<DataVector, SpatialDim, Frame>& coords,
    const double sigma_r) noexcept {
  tnsr::a<DataVector, SpatialDim, Frame> gauge_h{};
  tnsr::ab<DataVector, SpatialDim, Frame> d4_gauge_h{};
  GeneralizedHarmonic::gauges::damped_harmonic_rollon(
      make_not_null(&gauge_h), make_not_null(&d4_gauge_h), gauge_h_init,
      dgauge_h_init, lapse, shift, spacetime_unit_normal_one_form,
      sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric, pi,
      phi, t, coords, 1., 1.,
      1.,       // amp_coef_{L1, L2, S}
      4, 4, 4,  // exp_{L1, L2, S}
      t_start, sigma_t, sigma_r);
  return d4_gauge_h;
}

// Compare with Python implementation
template <size_t SpatialDim, typename Frame>
void test_with_python(const DataVector& used_for_size) noexcept {
  INFO("Test with python");
  CAPTURE(SpatialDim);
  CAPTURE(Frame{});
  // H_a
  {
    INFO("H_a");
    pypp::check_with_random_values<1>(
        static_cast<tnsr::a<DataVector, SpatialDim, Frame> (*)(
            const tnsr::a<DataVector, SpatialDim, Frame>&,
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
  {
    INFO("d4 H_a");
    pypp::check_with_random_values<1>(
        static_cast<tnsr::ab<DataVector, SpatialDim, Frame> (*)(
            const tnsr::a<DataVector, SpatialDim, Frame>&,
            const tnsr::ab<DataVector, SpatialDim, Frame>&,
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
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.DampedHarmonic",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  const DataVector used_for_size(4);

  test_options<Frame::Inertial>();

  {
    INFO("Test rollon function");
    test_rollon_function<1, Frame::Inertial>(used_for_size);
    test_rollon_function<2, Frame::Inertial>(used_for_size);
    test_rollon_function<3, Frame::Inertial>(used_for_size);

    test_rollon_function<1, Frame::Inertial>(1.);
    test_rollon_function<2, Frame::Inertial>(1.);
    test_rollon_function<3, Frame::Inertial>(1.);
  }

  {
    INFO("Compute source function");
    // Compare with Python implementation
    test_with_python<1, Frame::Inertial>(used_for_size);
    test_with_python<2, Frame::Inertial>(used_for_size);
    test_with_python<3, Frame::Inertial>(used_for_size);
  }
}
