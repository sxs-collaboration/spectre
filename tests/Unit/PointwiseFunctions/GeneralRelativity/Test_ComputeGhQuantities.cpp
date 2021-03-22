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
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/CovariantDerivOfExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/GaugeSource.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivOfDetSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivOfNormOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLowerShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpacetimeMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace Tags {
template <typename Tag, typename Dim, typename Frame, typename>
struct deriv;
}  // namespace Tags
/// \endcond

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
      static_cast<tnsr::a<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&, const Scalar<DataType>&,
          const tnsr::i<DataType, Dim, Frame::Inertial>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::iJ<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const Scalar<DataType>&,
          const tnsr::i<DataType, Dim, Frame::Inertial>&) noexcept>(
          &::GeneralizedHarmonic::gauge_source<Dim, Frame::Inertial, DataType>),
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
      spatial_metric_l.get(i, i) += 4.;
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
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size);
  // spacetime_deriv_of_det_spatial_metric
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::ii<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &::GeneralizedHarmonic::spacetime_deriv_of_det_spatial_metric<
              SpatialDim, Frame, DataType>),
      "GeneralRelativity.ComputeGhQuantities", "spacetime_deriv_detg",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void test_spacetime_metric_deriv_functions(
    const DataVector& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::aa<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &::GeneralizedHarmonic::time_derivative_of_spacetime_metric<
              SpatialDim, Frame, DataType>),
      "GeneralRelativity.ComputeGhQuantities", "gh_dt_spacetime_metric",
      {{{std::numeric_limits<double>::denorm_min(), 10.}}}, used_for_size);
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
  CHECK_ITERABLE_APPROX(get(shift_dot_dt_lower_shift_expected),
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

template <typename DataType, size_t SpatialDim, typename Frame>
void test_spacetime_derivative_of_spacetime_metric(
    const size_t num_pts) noexcept {
  CAPTURE(SpatialDim);
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 1.0);

  const auto lapse = make_with_random_values<Scalar<DataType>>(
      make_not_null(&generator), make_not_null(&distribution), num_pts);
  const auto shift =
      make_with_random_values<tnsr::I<DataType, SpatialDim, Frame>>(
          make_not_null(&generator), make_not_null(&distribution), num_pts);
  const auto pi =
      make_with_random_values<tnsr::aa<DataType, SpatialDim, Frame>>(
          make_not_null(&generator), make_not_null(&distribution), num_pts);
  const auto phi =
      make_with_random_values<tnsr::iaa<DataType, SpatialDim, Frame>>(
          make_not_null(&generator), make_not_null(&distribution), num_pts);

  const auto expected_dt_spacetime_metric =
      ::GeneralizedHarmonic::time_derivative_of_spacetime_metric(lapse, shift,
                                                                 pi, phi);
  tnsr::abb<DataType, SpatialDim, Frame> d4_spacetime_metric;
  ::GeneralizedHarmonic::spacetime_derivative_of_spacetime_metric(
      make_not_null(&d4_spacetime_metric), lapse, shift, pi, phi);
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        if (a == 0) {
          CHECK(d4_spacetime_metric.get(a, b, c) ==
                expected_dt_spacetime_metric.get(b, c));
        } else {
          CHECK(d4_spacetime_metric.get(a, b, c) == phi.get(a - 1, b, c));
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void test_cov_deriv_extrinsic_curvature(
    const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ijj<DataType, SpatialDim, Frame> (*)(
          const tnsr::ii<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::Ijj<DataType, SpatialDim, Frame>&,
          const tnsr::AA<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::ijaa<DataType, SpatialDim, Frame>&)>(
          &::GeneralizedHarmonic::covariant_deriv_of_extrinsic_curvature<
              SpatialDim, Frame, DataType>),
      "GeneralRelativity.ComputeGhQuantities",
      "covariant_deriv_extrinsic_curvture", {{{-1., 1.}}}, used_for_size);
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
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_cov_deriv_extrinsic_curvature,
                                    (1, 2, 3), (Frame::Grid, Frame::Inertial));

  const size_t num_pts = 5;
  const DataVector used_for_size(num_pts);
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

  test_spacetime_metric_deriv_functions<DataVector, 1, Frame::Grid>(
      used_for_size);
  test_spacetime_metric_deriv_functions<DataVector, 2, Frame::Grid>(
      used_for_size);
  test_spacetime_metric_deriv_functions<DataVector, 3, Frame::Grid>(
      used_for_size);
  test_spacetime_metric_deriv_functions<DataVector, 1, Frame::Inertial>(
      used_for_size);
  test_spacetime_metric_deriv_functions<DataVector, 2, Frame::Inertial>(
      used_for_size);
  test_spacetime_metric_deriv_functions<DataVector, 3, Frame::Inertial>(
      used_for_size);

  test_spacetime_derivative_of_spacetime_metric<DataVector, 1, Frame::Grid>(
      num_pts);
  test_spacetime_derivative_of_spacetime_metric<DataVector, 2, Frame::Grid>(
      num_pts);
  test_spacetime_derivative_of_spacetime_metric<DataVector, 3, Frame::Grid>(
      num_pts);
  test_spacetime_derivative_of_spacetime_metric<DataVector, 1, Frame::Inertial>(
      num_pts);
  test_spacetime_derivative_of_spacetime_metric<DataVector, 2, Frame::Inertial>(
      num_pts);
  test_spacetime_derivative_of_spacetime_metric<DataVector, 3, Frame::Inertial>(
      num_pts);

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

  // Check that compute items work correctly in the DataBox
  // First, check that the names are correct
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::TimeDerivSpatialMetricCompute<
          3, Frame::Inertial>>("dt(SpatialMetric)");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::TimeDerivLapseCompute<3, Frame::Inertial>>(
      "dt(Lapse)");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::TimeDerivShiftCompute<3, Frame::Inertial>>(
      "dt(Shift)");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::DerivSpatialMetricCompute<3, Frame::Inertial>>(
      "deriv(SpatialMetric)");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::DerivLapseCompute<3, Frame::Inertial>>(
      "deriv(Lapse)");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::DerivShiftCompute<3, Frame::Inertial>>(
      "deriv(Shift)");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::PhiCompute<3, Frame::Inertial>>("Phi");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::PiCompute<3, Frame::Inertial>>("Pi");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::ExtrinsicCurvatureCompute<3, Frame::Inertial>>(
      "ExtrinsicCurvature");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::TraceExtrinsicCurvatureCompute<
          3, Frame::Inertial>>("TraceExtrinsicCurvature");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma0Compute<
          3, Frame::Inertial>>("ConstraintGamma0");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1Compute<
          3, Frame::Inertial>>("ConstraintGamma1");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2Compute<
          3, Frame::Inertial>>("ConstraintGamma2");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<3,
                                                             Frame::Inertial>>(
      "SpacetimeDerivGaugeH");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::GaugeHImplicitFrom3p1QuantitiesCompute<
          3, Frame::Inertial>>("GaugeH");

  // Check that the compute items return the correct values
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-0.1, 0.1);
  std::uniform_real_distribution<> dist_positive(1., 2.);

  const auto spacetime_metric = [&]() {
    auto spacetime_metric_l =
        make_with_random_values<tnsr::aa<DataVector, 3, Frame::Inertial>>(
            make_not_null(&generator), make_not_null(&distribution),
            used_for_size);
    // Make sure spacetime_metric isn't singular.
    get<0, 0>(spacetime_metric_l) += -1.;
    for (size_t i = 1; i <= 3; ++i) {
      spacetime_metric_l.get(i, i) += 1.;
    }
    return spacetime_metric_l;
  }();

  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto det_and_inv_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto shift =
      gr::shift(spacetime_metric, det_and_inv_spatial_metric.second);
  const auto lapse = gr::lapse(shift, spacetime_metric);

  const auto expected_dt_spatial_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const auto expected_dt_shift =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const auto expected_dt_lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  const auto expected_deriv_lapse =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&dist_positive),
          used_for_size);
  const auto expected_deriv_shift =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const auto expected_deriv_spatial_metric =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);
  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);

  const auto expected_phi = GeneralizedHarmonic::phi(
      lapse, expected_deriv_lapse, shift, expected_deriv_shift, spatial_metric,
      expected_deriv_spatial_metric);
  const auto expected_pi = GeneralizedHarmonic::pi(
      lapse, expected_dt_lapse, shift, expected_dt_shift, spatial_metric,
      expected_dt_spatial_metric, expected_phi);
  const auto expected_extrinsic_curvature =
      GeneralizedHarmonic::extrinsic_curvature(spacetime_normal_vector,
                                               expected_pi, expected_phi);
  const auto expected_trace_extrinsic_curvature =
      trace(expected_extrinsic_curvature, inverse_spatial_metric);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::Lapse<DataVector>,
          gr::Tags::SpacetimeNormalVector<3, Frame::Inertial, DataVector>,
          gr::Tags::InverseSpacetimeMetric<3, Frame::Inertial, DataVector>,
          GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>,
      db::AddComputeTags<
          GeneralizedHarmonic::Tags::DerivSpatialMetricCompute<3,
                                                               Frame::Inertial>,
          GeneralizedHarmonic::Tags::DerivLapseCompute<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::DerivShiftCompute<3, Frame::Inertial>>>(
      lapse, spacetime_normal_vector, inverse_spacetime_metric, expected_phi);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(db::get<::Tags::deriv<
                     gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                     tmpl::size_t<3>, Frame::Inertial>>(box)),
      expected_deriv_spatial_metric);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(db::get<::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                       tmpl::size_t<3>, Frame::Inertial>>(box)),
      expected_deriv_lapse);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(
          db::get<::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                                tmpl::size_t<3>, Frame::Inertial>>(box)),
      expected_deriv_shift);

  const auto other_box = db::create<
      db::AddSimpleTags<
          gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<3, Frame::Inertial, DataVector>,
          gr::Tags::SpacetimeNormalVector<3, Frame::Inertial, DataVector>,
          gr::Tags::InverseSpacetimeMetric<3, Frame::Inertial, DataVector>,
          gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
          GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>,
      db::AddComputeTags<
          GeneralizedHarmonic::Tags::TimeDerivSpatialMetricCompute<
              3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::TimeDerivLapseCompute<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::TimeDerivShiftCompute<3,
                                                           Frame::Inertial>>>(
      lapse, shift, spacetime_normal_vector, inverse_spacetime_metric,
      inverse_spatial_metric, expected_phi, expected_pi);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(db::get<::Tags::dt<
                     gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
          other_box)),
      expected_dt_spatial_metric);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(db::get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(other_box)),
      expected_dt_lapse);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(
          db::get<::Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataVector>>>(
              other_box)),
      expected_dt_shift);

  const auto ghvars_box = db::create<
      db::AddSimpleTags<
          gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
          gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<3, Frame::Inertial, DataVector>,
          gr::Tags::SpacetimeNormalVector<3, Frame::Inertial, DataVector>,
          gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
          ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>,
          ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>,
          ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>,
          ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>,
          ::Tags::dt<gr::Tags::Lapse<DataVector>>,
          ::Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataVector>>>,
      db::AddComputeTags<
          GeneralizedHarmonic::Tags::PhiCompute<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::PiCompute<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::ExtrinsicCurvatureCompute<3,
                                                               Frame::Inertial>,
          GeneralizedHarmonic::Tags::TraceExtrinsicCurvatureCompute<
              3, Frame::Inertial>>>(
      spatial_metric, lapse, shift,
      spacetime_normal_vector, inverse_spatial_metric,
      expected_deriv_spatial_metric, expected_deriv_lapse, expected_deriv_shift,
      expected_dt_spatial_metric, expected_dt_lapse, expected_dt_shift);

  CHECK(db::get<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(
            ghvars_box) == expected_phi);
  CHECK(db::get<GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(
            ghvars_box) == expected_pi);
  CHECK(db::get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>(
            ghvars_box) == expected_extrinsic_curvature);
  CHECK(db::get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(ghvars_box) ==
        expected_trace_extrinsic_curvature);
}
