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
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedWaveHelpers.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
/// \endcond

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
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

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
  GeneralizedHarmonic::gauges::damped_harmonic(
      make_not_null(&gauge_h), make_not_null(&d4_gauge_h), gauge_h_init,
      dgauge_h_init, lapse, shift, spacetime_unit_normal_one_form,
      sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric, pi,
      phi, t, coords, 1., 1.,
      1.,                // amp_coef_{L1, L2, S}
      4, 4, 4,           // exp_{L1, L2, S}
      t_start, sigma_t,  // _h_init
      t_start, sigma_t,  // _L1
      t_start, sigma_t,  // _L2
      t_start, sigma_t,  // _S
      sigma_r);
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
  GeneralizedHarmonic::gauges::damped_harmonic(
      make_not_null(&gauge_h), make_not_null(&d4_gauge_h), gauge_h_init,
      dgauge_h_init, lapse, shift, spacetime_unit_normal_one_form,
      sqrt_det_spatial_metric, inverse_spatial_metric, spacetime_metric, pi,
      phi, t, coords, 1., 1.,
      1.,                // amp_coef_{L1, L2, S}
      4, 4, 4,           // exp_{L1, L2, S}
      t_start, sigma_t,  // _h_init
      t_start, sigma_t,  // _L1
      t_start, sigma_t,  // _L2
      t_start, sigma_t,  // _S
      sigma_r);
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
  const auto spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto spacetime_unit_normal_one_form =
      gr::spacetime_normal_one_form<SpatialDim, Frame::Inertial, DataVector>(
          lapse);
  const auto spacetime_unit_normal =
      gr::spacetime_normal_vector<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift);
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& det_spatial_metric = det_and_inv.first;
  const auto& inverse_spatial_metric = det_and_inv.second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric<SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift, inverse_spatial_metric);
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
  const auto gauge_h_init =
      make_with_random_values<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&pdist), x);
  const auto d4_gauge_h_init = make_with_random_values<
      tnsr::ab<DataVector, SpatialDim, Frame::Inertial>>(
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
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::gauges::DampedHarmonicCompute<3, Frame::Inertial>>(
      "DampedHarmonicCompute");

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
          GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>, ::Tags::Time,
          domain::Tags::Coordinates<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::GaugeHRollOnStartTime,
          GeneralizedHarmonic::Tags::GaugeHRollOnTimeWindow,
          GeneralizedHarmonic::Tags::GaugeHSpatialWeightDecayWidth<
              Frame::Inertial>>,
      db::AddComputeTags<GeneralizedHarmonic::gauges::DampedHarmonicCompute<
          3, Frame::Inertial>>>(gauge_h_init, d4_gauge_h_init, lapse, shift,
                                spacetime_unit_normal_one_form,
                                sqrt_det_spatial_metric, inverse_spatial_metric,
                                spacetime_metric, pi, phi, t, x, t_start_S,
                                sigma_t_S, r_max);

  // Verify that locally computed H_a matches the same obtained through its
  // ComputeTag from databox
  CHECK(db::get<GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>>(box) ==
        gauge_h_expected);
  CHECK(
      db::get<
          GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<3, Frame::Inertial>>(
          box) == d4_gauge_h_expected);
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

  {
    INFO("Compute tags");
    const size_t grid_size = 8;
    const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
    const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};
    MAKE_GENERATOR(generator);

    // ComputeTag tests
    test_damped_harmonic_compute_tags(grid_size, lower_bound, upper_bound,
                                      generator);
  }
}
