// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t Dim, typename DataType>
void test_compute_phi(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iaa<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&,
          const tnsr::i<DataType, Dim, Frame::Inertial>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::iJ<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::ijj<DataType, Dim, Frame::Inertial>&)>(
          &::GeneralizedHarmonic::phi<Dim, Frame::Inertial, DataType>),
      "GeneralRelativity.ComputeGhQuantities", "phi", {{{-10., 10.}}},
      used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_pi(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::aa<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&, const Scalar<DataType>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::iaa<DataType, Dim, Frame::Inertial>&)>(
          &::GeneralizedHarmonic::pi<Dim, Frame::Inertial, DataType>),
      "GeneralRelativity.ComputeGhQuantities", "pi", {{{-10., 10.}}},
      used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_gauge_source(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &GeneralizedHarmonic::gauge_source<Dim, Frame::Inertial, DataType>,
      "GeneralRelativity.ComputeGhQuantities", "gauge_source", {{{-10., 10.}}},
      used_for_size, 1.e-11);
}

template <size_t Dim, typename T>
void test_compute_extrinsic_curvature_and_deriv_metric(const T& used_for_size) {
  // Set up random values for lapse, shift, spatial_metric,
  // and their derivatives.
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  std::uniform_real_distribution<> dist_positive(1., 2.);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);
  const auto nn_dist_positive = make_not_null(&dist_positive);

  const auto lapse = make_with_random_values<Scalar<T>>(
      nn_generator, nn_dist_positive, used_for_size);
  const auto shift = make_with_random_values<tnsr::I<T, Dim>>(
      nn_generator, nn_dist, used_for_size);
  const auto spatial_metric = [&]() {
    auto spatial_metric_l = make_with_random_values<tnsr::ii<T, Dim>>(
        nn_generator, nn_dist, used_for_size);
    // Make sure spatial_metric isn't singular by adding
    // large enough positive diagonal values.
    for (size_t i = 0; i < Dim; ++i) {
      spatial_metric_l.get(i, i) += 4.0;
    }
    return spatial_metric_l;
  }();
  const auto dt_lapse = make_with_random_values<Scalar<T>>(
      nn_generator, nn_dist_positive, used_for_size);
  const auto deriv_lapse = make_with_random_values<tnsr::i<T, Dim>>(
      nn_generator, nn_dist_positive, used_for_size);
  const auto dt_shift = make_with_random_values<tnsr::I<T, Dim>>(
      nn_generator, nn_dist, used_for_size);
  const auto deriv_shift = make_with_random_values<tnsr::iJ<T, Dim>>(
      nn_generator, nn_dist, used_for_size);
  const auto deriv_spatial_metric = make_with_random_values<tnsr::ijj<T, Dim>>(
      nn_generator, nn_dist, used_for_size);
  const auto dt_spatial_metric = make_with_random_values<tnsr::ii<T, Dim>>(
      nn_generator, nn_dist, used_for_size);

  // Make extrinsic curvature, spacetime_normal_vector, and generalized
  // harmonic pi,psi variables in a way that is already independently tested.
  const auto extrinsic_curvature =
      gr::extrinsic_curvature(lapse, shift, deriv_shift, spatial_metric,
                              dt_spatial_metric, deriv_spatial_metric);
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);
  const auto phi =
      GeneralizedHarmonic::phi(lapse, deriv_lapse, shift, deriv_shift,
                               spatial_metric, deriv_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);

  // Compute extrinsic curvature and deriv_spatial_metric from generalized
  // harmonic variables and make sure we get the same result.
  const auto extrinsic_curvature_test =
      GeneralizedHarmonic::extrinsic_curvature(spacetime_normal_vector, pi,
                                               phi);
  const auto deriv_spatial_metric_test =
      GeneralizedHarmonic::deriv_spatial_metric(phi);

  CHECK_ITERABLE_APPROX(extrinsic_curvature, extrinsic_curvature_test);
  CHECK_ITERABLE_APPROX(deriv_spatial_metric, deriv_spatial_metric_test);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void test_lapse_deriv_functions(const DataVector& used_for_size) noexcept {
  // spatial_deriv_of_lapse
  pypp::check_with_random_values<1>(
      static_cast<tnsr::i<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &::GeneralizedHarmonic::spatial_deriv_of_lapse<SpatialDim, Frame,
                                                         DataType>),
      "GeneralRelativity.ComputeGhQuantities", "deriv_lapse",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size);
  // time_deriv_of_lapse
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(
          const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&)>(
          &::GeneralizedHarmonic::time_deriv_of_lapse<SpatialDim, Frame,
                                                      DataType>),
      "GeneralRelativity.ComputeGhQuantities", "dt_lapse",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size,
      1.e-11);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void test_gij_deriv_functions(const DataVector& used_for_size) noexcept {
  // time_deriv_of_spatial_metric
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ii<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&)>(
          &::GeneralizedHarmonic::time_deriv_of_spatial_metric<
              SpatialDim, Frame, DataType>),
      "GeneralRelativity.ComputeGhQuantities", "dt_spatial_metric",
      {{{std::numeric_limits<double>::denorm_min(), 10.0}}}, used_for_size);
  // spacetime_deriv_of_det_spatial_metric
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::ii<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &::GeneralizedHarmonic::spacetime_deriv_of_det_spatial_metric<
              SpatialDim, Frame, DataType>),
      "GeneralRelativity.ComputeGhQuantities", "spacetime_deriv_detg",
      {{{std::numeric_limits<double>::denorm_min(), 10.0}}}, used_for_size);
}

// Test computation of derivs of lapse by comparing to Kerr-Schild
template <typename Solution>
void test_lapse_deriv_functions_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Setup grid
  const size_t spatial_dim = 3;
  Mesh<spatial_dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
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
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& dt_lapse_expected = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& d_lapse_expected =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<spatial_dim>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<spatial_dim>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<spatial_dim>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<spatial_dim>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients
  const auto phi =
      GeneralizedHarmonic::phi(lapse, d_lapse_expected, shift, d_shift,
                               spatial_metric, d_spatial_metric);
  const auto pi =
      GeneralizedHarmonic::pi(lapse, dt_lapse_expected, shift, dt_shift,
                              spatial_metric, dt_spatial_metric, phi);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);

  // Check that locally computed derivs match returned ones
  const auto dt_lapse =
      GeneralizedHarmonic::time_deriv_of_lapse<spatial_dim, Frame::Inertial,
                                               DataVector>(
          lapse, shift, normal_vector, phi, pi);

  const auto d_lapse =
      GeneralizedHarmonic::spatial_deriv_of_lapse<spatial_dim, Frame::Inertial,
                                                  DataVector>(
          lapse, normal_vector, phi);

  CHECK_ITERABLE_APPROX(dt_lapse_expected, dt_lapse);
  CHECK_ITERABLE_APPROX(d_lapse_expected, d_lapse);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void test_shift_deriv_functions(const DataVector& used_for_size) noexcept {
  // spatial_deriv_of_shift
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iJ<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::AA<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &::GeneralizedHarmonic::spatial_deriv_of_shift<SpatialDim, Frame,
                                                         DataType>),
      "GeneralRelativity.ComputeGhQuantities", "deriv_shift",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size);
  // time_deriv_of_shift
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>& lapse,
          const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&)>(
          &::GeneralizedHarmonic::time_deriv_of_shift<SpatialDim, Frame,
                                                      DataType>),
      "GeneralRelativity.ComputeGhQuantities", "dt_shift",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size,
      1.e-10);
  // time_deriv_of_lower_shift
  pypp::check_with_random_values<1>(
      static_cast<tnsr::i<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::ii<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&)>(
          &::GeneralizedHarmonic::time_deriv_of_lower_shift<SpatialDim, Frame,
                                                            DataType>),
      "GeneralRelativity.ComputeGhQuantities", "dt_lower_shift",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size,
      5.e-9);
  // spacetime_deriv_of_norm_of_shift
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::ii<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::AA<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&)>(
          &::GeneralizedHarmonic::spacetime_deriv_of_norm_of_shift<
              SpatialDim, Frame, DataType>),
      "GeneralRelativity.ComputeGhQuantities", "spacetime_deriv_norm_shift",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size,
      1.e-10);
}

// Test computation of derivs of shift by comparing to Kerr-Schild
template <typename Solution>
void test_shift_deriv_functions_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
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
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<SpatialDim>>(vars);
  const auto& d_shift_expected =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift_expected =
      get<Tags::dt<gr::Tags::Shift<SpatialDim>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<SpatialDim>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<SpatialDim>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto phi =
      GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift_expected,
                               spatial_metric, d_spatial_metric);
  const auto pi =
      GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift_expected,
                              spatial_metric, dt_spatial_metric, phi);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);
  const auto lower_shift = raise_or_lower_index(shift, spatial_metric);

  // Get d4_spacetime_metric for d4_norm_shift
  const auto d4_spacetime_metric = gr::derivatives_of_spacetime_metric(
      lapse, dt_lapse, d_lapse, shift, dt_shift_expected, d_shift_expected,
      spatial_metric, dt_spatial_metric, d_spatial_metric);

  // Compute d4_norm_shift = d4_spacetime_metric_00 + 2 lapse d4_lapse
  auto d4_norm_of_shift_expected =
      make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  get<0>(d4_norm_of_shift_expected) =
      d4_spacetime_metric.get(0, 0, 0) + 2. * get(lapse) * get(dt_lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_norm_of_shift_expected.get(1 + i) =
        d4_spacetime_metric.get(i + 1, 0, 0) + 2. * get(lapse) * d_lapse.get(i);
  }

  // Compute dt_norm_of_shift - lower_shift_dot_dt_shift =
  // shift_dot_dt_lower_shift
  auto shift_dot_dt_lower_shift_expected =
      make_with_value<Scalar<DataVector>>(x, 0.);
  auto lower_shift_dot_dt_shift = make_with_value<Scalar<DataVector>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    get(lower_shift_dot_dt_shift) +=
        lower_shift.get(i) * dt_shift_expected.get(i);
  }
  get(shift_dot_dt_lower_shift_expected) =
      get<0>(d4_norm_of_shift_expected) - get(lower_shift_dot_dt_shift);

  // Check that locally computed derivs match returned ones
  const auto dt_shift =
      GeneralizedHarmonic::time_deriv_of_shift<SpatialDim, Frame::Inertial,
                                               DataVector>(
          lapse, shift, inverse_spatial_metric, normal_vector, phi, pi);

  const auto d_shift =
      GeneralizedHarmonic::spatial_deriv_of_shift<SpatialDim, Frame::Inertial,
                                                  DataVector>(
          lapse, inverse_spacetime_metric, normal_vector, phi);

  const auto d4_norm_shift =
      GeneralizedHarmonic::spacetime_deriv_of_norm_of_shift<
          SpatialDim, Frame::Inertial, DataVector>(
          lapse, shift, spatial_metric, inverse_spatial_metric,
          inverse_spacetime_metric, normal_vector, phi, pi);

  const auto dt_lower_shift = GeneralizedHarmonic::time_deriv_of_lower_shift<
      SpatialDim, Frame::Inertial, DataVector>(lapse, shift, spatial_metric,
                                               normal_vector, phi, pi);

  auto shift_dot_dt_lower_shift = make_with_value<Scalar<DataVector>>(x, 0.);
  for (size_t i = 0; i < SpatialDim; ++i) {
    get(shift_dot_dt_lower_shift) += shift.get(i) * dt_lower_shift.get(i);
  }

  CHECK_ITERABLE_APPROX(dt_shift_expected, dt_shift);
  CHECK_ITERABLE_APPROX(d_shift_expected, d_shift);
  CHECK_ITERABLE_APPROX(d4_norm_of_shift_expected, d4_norm_shift);
  CHECK(get(shift_dot_dt_lower_shift_expected) ==
        get(shift_dot_dt_lower_shift));
}

// Test computation of derivs of spatial metric by comparing to Kerr-Schild
template <typename Solution>
void test_gij_deriv_functions_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
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
  const double t = std::numeric_limits<double>::signaling_NaN();
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
  const auto& dt_spatial_metric_expected =
      get<Tags::dt<gr::Tags::SpatialMetric<SpatialDim>>>(vars);
  const auto& d_spatial_metric_expected =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);
  // Get ingredients
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto inverse_spatial_metric = det_and_inv.second;
  const auto det_spatial_metric = det_and_inv.first;
  const Scalar<DataVector> sqrt_det_spatial_metric{
      sqrt(get(det_spatial_metric))};
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto d4_spacetime_metric = gr::derivatives_of_spacetime_metric(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, spatial_metric,
      dt_spatial_metric_expected, d_spatial_metric_expected);
  const auto christoffel_first =
      gr::christoffel_first_kind(d4_spacetime_metric);
  const auto phi =
      GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift, spatial_metric,
                               d_spatial_metric_expected);
  const auto pi =
      GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
                              dt_spatial_metric_expected, phi);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);

  // Get spacetime deriv of Det[g]:
  // \partial_a g = -2 g \partial_a \alpha / \alpha + 2 g \Gamma^b_{a b}
  auto d4_g_expected =
      make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  // \Gamma^b_{a b} = \psi^{bc} \Gamma_{bac}
  auto inv_psi_dot_christoffel =
      make_with_value<tnsr::a<DataVector, SpatialDim, Frame::Inertial>>(x, 0.);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        inv_psi_dot_christoffel.get(a) +=
            inverse_spacetime_metric.get(b, c) * christoffel_first.get(b, a, c);
      }
    }
  }
  const auto pre_factor1 = -2. * get(det_spatial_metric) / get(lapse);
  get<0>(d4_g_expected) =
      pre_factor1 * get(dt_lapse) +
      2. * get(det_spatial_metric) * get<0>(inv_psi_dot_christoffel);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_g_expected.get(1 + i) =
        pre_factor1 * d_lapse.get(i) +
        2. * get(det_spatial_metric) * inv_psi_dot_christoffel.get(i + 1);
  }

  // Check that locally computed derivs match returned ones
  const auto dt_gij = GeneralizedHarmonic::time_deriv_of_spatial_metric<
      SpatialDim, Frame::Inertial, DataVector>(lapse, shift, phi, pi);
  const auto d_gij =
      GeneralizedHarmonic::deriv_spatial_metric<SpatialDim, Frame::Inertial,
                                                DataVector>(phi);
  const auto d4_g = GeneralizedHarmonic::spacetime_deriv_of_det_spatial_metric<
      SpatialDim, Frame::Inertial, DataVector>(sqrt_det_spatial_metric,
                                               inverse_spatial_metric,
                                               dt_spatial_metric_expected, phi);

  CHECK_ITERABLE_APPROX(dt_spatial_metric_expected, dt_gij);
  CHECK_ITERABLE_APPROX(d_spatial_metric_expected, d_gij);
  CHECK_ITERABLE_APPROX(d4_g_expected, d4_g);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.GhQuantities",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env("PointwiseFunctions/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_phi, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_pi, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_gauge_source, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(
      test_compute_extrinsic_curvature_and_deriv_metric, (1, 2, 3));

  const DataVector used_for_size(20);
  test_lapse_deriv_functions<DataVector, 1, Frame::Grid>(used_for_size);
  test_lapse_deriv_functions<DataVector, 2, Frame::Grid>(used_for_size);
  test_lapse_deriv_functions<DataVector, 3, Frame::Grid>(used_for_size);
  test_lapse_deriv_functions<DataVector, 1, Frame::Inertial>(used_for_size);
  test_lapse_deriv_functions<DataVector, 2, Frame::Inertial>(used_for_size);
  test_lapse_deriv_functions<DataVector, 3, Frame::Inertial>(used_for_size);

  test_gij_deriv_functions<DataVector, 1, Frame::Grid>(used_for_size);
  test_gij_deriv_functions<DataVector, 2, Frame::Grid>(used_for_size);
  test_gij_deriv_functions<DataVector, 3, Frame::Grid>(used_for_size);
  test_gij_deriv_functions<DataVector, 1, Frame::Inertial>(used_for_size);
  test_gij_deriv_functions<DataVector, 2, Frame::Inertial>(used_for_size);
  test_gij_deriv_functions<DataVector, 3, Frame::Inertial>(used_for_size);

  test_shift_deriv_functions<DataVector, 1, Frame::Grid>(used_for_size);
  test_shift_deriv_functions<DataVector, 2, Frame::Grid>(used_for_size);
  test_shift_deriv_functions<DataVector, 3, Frame::Grid>(used_for_size);
  test_shift_deriv_functions<DataVector, 1, Frame::Inertial>(used_for_size);
  test_shift_deriv_functions<DataVector, 2, Frame::Inertial>(used_for_size);
  test_shift_deriv_functions<DataVector, 3, Frame::Inertial>(used_for_size);

  const double mass = 2.;
  const std::array<double, 3> spin{{0.3, 0.5, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};

  test_lapse_deriv_functions_analytic(solution, grid_size, lower_bound,
                                      upper_bound);
  test_gij_deriv_functions_analytic(solution, grid_size, lower_bound,
                                    upper_bound);
  test_shift_deriv_functions_analytic(solution, grid_size, lower_bound,
                                      upper_bound);
}
