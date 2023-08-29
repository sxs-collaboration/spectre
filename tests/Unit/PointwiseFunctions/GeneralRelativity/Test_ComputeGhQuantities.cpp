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
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/CovariantDerivOfExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/GaugeSource.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Ricci.hpp"
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
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpacetimeMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace Tags {
template <typename Tag, typename Dim, typename Frame, typename>
struct deriv;
}  // namespace Tags

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
          &::gh::phi<DataType, Dim, Frame::Inertial>),
      "GeneralRelativity.ComputeGhQuantities", "phi", {{{-1., 1.}}},
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
          &::gh::pi<DataType, Dim, Frame::Inertial>),
      "GeneralRelativity.ComputeGhQuantities", "pi", {{{-1., 1.}}},
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
          const tnsr::i<DataType, Dim, Frame::Inertial>&)>(
          &::gh::gauge_source<DataType, Dim, Frame::Inertial>),
      "GeneralRelativity.ComputeGhQuantities", "gauge_source", {{{-1., 1.}}},
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
  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<T, Dim, Frame::Inertial>(lapse);
  const auto phi = gh::phi(lapse, deriv_lapse, shift, deriv_shift,
                           spatial_metric, deriv_spatial_metric);
  const auto pi = gh::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
                         dt_spatial_metric, phi);

  // Compute extrinsic curvature and deriv_spatial_metric from generalized
  // harmonic variables and make sure we get the same result.
  const auto extrinsic_curvature_test =
      gh::extrinsic_curvature(spacetime_normal_vector, pi, phi);
  const auto deriv_spatial_metric_test = gh::deriv_spatial_metric(phi);

  CHECK_ITERABLE_APPROX(extrinsic_curvature, extrinsic_curvature_test);
  CHECK_ITERABLE_APPROX(deriv_spatial_metric, deriv_spatial_metric_test);

  // Compute Christoffel symbol of the 2nd kind in two different ways
  // (the gr one already tested independently) and make sure we get
  // the same result.
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto christoffel_second_kind =
      gh::christoffel_second_kind(phi, inverse_spatial_metric);
  const auto christoffel_second_kind_test = raise_or_lower_first_index(
      gr::christoffel_first_kind(deriv_spatial_metric), inverse_spatial_metric);

  CHECK_ITERABLE_APPROX(christoffel_second_kind, christoffel_second_kind_test);

  // Test Christoffel trace
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto trace_christoffel = gh::trace_christoffel(
      spacetime_normal_one_form, spacetime_normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi);
  tnsr::abb<T, Dim, Frame::Inertial> d4_spacetime_metric;
  ::gh::spacetime_derivative_of_spacetime_metric(
      make_not_null(&d4_spacetime_metric), lapse, shift, pi, phi);
  const auto expected_trace_christoffel =
      trace_last_indices(gr::christoffel_first_kind(d4_spacetime_metric),
                         inverse_spacetime_metric);
  CHECK_ITERABLE_APPROX(trace_christoffel, expected_trace_christoffel);
  tnsr::a<T, Dim, Frame::Inertial> (*f_trace_christoffel)(
      const tnsr::a<T, Dim, Frame::Inertial>&,
      const tnsr::A<T, Dim, Frame::Inertial>&,
      const tnsr::II<T, Dim, Frame::Inertial>&,
      const tnsr::AA<T, Dim, Frame::Inertial>&,
      const tnsr::aa<T, Dim, Frame::Inertial>&,
      const tnsr::iaa<T, Dim, Frame::Inertial>&) =
      &gh::trace_christoffel<T, Dim, Frame::Inertial>;
  pypp::check_with_random_values<1>(
      f_trace_christoffel, "GeneralRelativity.ComputeGhQuantities",
      "trace_christoffel", {{{-1., 1.}}}, used_for_size, 1.e-11);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void test_lapse_deriv_functions(const DataVector& used_for_size) {
  // spatial_deriv_of_lapse
  pypp::check_with_random_values<1>(
      static_cast<tnsr::i<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &::gh::spatial_deriv_of_lapse<DataType, SpatialDim, Frame>),
      "GeneralRelativity.ComputeGhQuantities", "deriv_lapse",
      {{{std::numeric_limits<double>::denorm_min(), 1.}}}, used_for_size);
  // time_deriv_of_lapse
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(
          const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&)>(
          &::gh::time_deriv_of_lapse<DataType, SpatialDim, Frame>),
      "GeneralRelativity.ComputeGhQuantities", "dt_lapse",
      {{{std::numeric_limits<double>::denorm_min(), 1.}}}, used_for_size,
      1.e-11);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void test_gij_deriv_functions(const DataVector& used_for_size) {
  // time_deriv_of_spatial_metric
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ii<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&)>(
          &::gh::time_deriv_of_spatial_metric<DataType, SpatialDim, Frame>),
      "GeneralRelativity.ComputeGhQuantities", "dt_spatial_metric",
      {{{std::numeric_limits<double>::denorm_min(), 1.}}}, used_for_size);
  // spacetime_deriv_of_det_spatial_metric
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::ii<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &::gh::spacetime_deriv_of_det_spatial_metric<DataType, SpatialDim,
                                                       Frame>),
      "GeneralRelativity.ComputeGhQuantities", "spacetime_deriv_detg",
      {{{std::numeric_limits<double>::denorm_min(), 1.}}}, used_for_size);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void test_spacetime_metric_deriv_functions(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::aa<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &::gh::time_derivative_of_spacetime_metric<DataType, SpatialDim,
                                                     Frame>),
      "GeneralRelativity.ComputeGhQuantities", "gh_dt_spacetime_metric",
      {{{std::numeric_limits<double>::denorm_min(), 1.}}}, used_for_size);
}

// Test computation of derivs of lapse by comparing to Kerr-Schild
template <typename Solution>
void test_lapse_deriv_functions_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) {
  // Setup grid
  const size_t spatial_dim = 3;
  Mesh<spatial_dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
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
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto& dt_lapse_expected =
      get<Tags::dt<gr::Tags::Lapse<DataVector>>>(vars);
  const auto& d_lapse_expected =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<DataVector, spatial_dim>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift =
      get<Tags::dt<gr::Tags::Shift<DataVector, spatial_dim>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, spatial_dim>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataVector, spatial_dim>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients
  const auto phi = gh::phi(lapse, d_lapse_expected, shift, d_shift,
                           spatial_metric, d_spatial_metric);
  const auto pi = gh::pi(lapse, dt_lapse_expected, shift, dt_shift,
                         spatial_metric, dt_spatial_metric, phi);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);

  // Check that locally computed derivs match returned ones
  const auto dt_lapse =
      gh::time_deriv_of_lapse(lapse, shift, normal_vector, phi, pi);

  const auto d_lapse = gh::spatial_deriv_of_lapse(lapse, normal_vector, phi);

  CHECK_ITERABLE_APPROX(dt_lapse_expected, dt_lapse);
  CHECK_ITERABLE_APPROX(d_lapse_expected, d_lapse);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void test_shift_deriv_functions(const DataVector& used_for_size) {
  // spatial_deriv_of_shift
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iJ<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::AA<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &::gh::spatial_deriv_of_shift<DataType, SpatialDim, Frame>),
      "GeneralRelativity.ComputeGhQuantities", "deriv_shift",
      {{{std::numeric_limits<double>::denorm_min(), 1.}}}, used_for_size);
  // time_deriv_of_shift
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>& lapse,
          const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&)>(
          &::gh::time_deriv_of_shift<DataType, SpatialDim, Frame>),
      "GeneralRelativity.ComputeGhQuantities", "dt_shift",
      {{{std::numeric_limits<double>::denorm_min(), 1.}}}, used_for_size,
      1.e-10);
  // time_deriv_of_lower_shift
  pypp::check_with_random_values<1>(
      static_cast<tnsr::i<DataType, SpatialDim, Frame> (*)(
          const Scalar<DataType>&, const tnsr::I<DataType, SpatialDim, Frame>&,
          const tnsr::ii<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&)>(
          &::gh::time_deriv_of_lower_shift<DataType, SpatialDim, Frame>),
      "GeneralRelativity.ComputeGhQuantities", "dt_lower_shift",
      {{{std::numeric_limits<double>::denorm_min(), 1.}}}, used_for_size,
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
          &::gh::spacetime_deriv_of_norm_of_shift<DataType, SpatialDim, Frame>),
      "GeneralRelativity.ComputeGhQuantities", "spacetime_deriv_norm_shift",
      {{{std::numeric_limits<double>::denorm_min(), 1.}}}, used_for_size,
      1.e-10);
}

// Test computation of derivs of shift by comparing to Kerr-Schild
template <typename Solution>
void test_shift_deriv_functions_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) {
  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
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
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataVector>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<DataVector, SpatialDim>>(vars);
  const auto& d_shift_expected =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift_expected =
      get<Tags::dt<gr::Tags::Shift<DataVector, SpatialDim>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataVector, SpatialDim>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto phi = gh::phi(lapse, d_lapse, shift, d_shift_expected,
                           spatial_metric, d_spatial_metric);
  const auto pi = gh::pi(lapse, dt_lapse, shift, dt_shift_expected,
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
  const auto dt_shift = gh::time_deriv_of_shift(
      lapse, shift, inverse_spatial_metric, normal_vector, phi, pi);

  const auto d_shift = gh::spatial_deriv_of_shift(
      lapse, inverse_spacetime_metric, normal_vector, phi);

  const auto d4_norm_shift = gh::spacetime_deriv_of_norm_of_shift(
      lapse, shift, spatial_metric, inverse_spatial_metric,
      inverse_spacetime_metric, normal_vector, phi, pi);

  const auto dt_lower_shift = gh::time_deriv_of_lower_shift(
      lapse, shift, spatial_metric, normal_vector, phi, pi);

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
    const std::array<double, 3>& upper_bound) {
  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
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
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataVector>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<DataVector, SpatialDim>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift =
      get<Tags::dt<gr::Tags::Shift<DataVector, SpatialDim>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim>>(vars);
  const auto& dt_spatial_metric_expected =
      get<Tags::dt<gr::Tags::SpatialMetric<DataVector, SpatialDim>>>(vars);
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
  const auto phi = gh::phi(lapse, d_lapse, shift, d_shift, spatial_metric,
                           d_spatial_metric_expected);
  const auto pi = gh::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
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
  const auto dt_gij = gh::time_deriv_of_spatial_metric(lapse, shift, phi, pi);
  const auto d_gij = gh::deriv_spatial_metric(phi);
  const auto d4_g = gh::spacetime_deriv_of_det_spatial_metric(
      sqrt_det_spatial_metric, inverse_spatial_metric,
      dt_spatial_metric_expected, phi);

  CHECK_ITERABLE_APPROX(dt_spatial_metric_expected, dt_gij);
  CHECK_ITERABLE_APPROX(d_spatial_metric_expected, d_gij);
  CHECK_ITERABLE_APPROX(d4_g_expected, d4_g);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void test_spacetime_derivative_of_spacetime_metric(const size_t num_pts) {
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
      ::gh::time_derivative_of_spacetime_metric(lapse, shift, pi, phi);
  tnsr::abb<DataType, SpatialDim, Frame> d4_spacetime_metric;
  ::gh::spacetime_derivative_of_spacetime_metric(
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
void test_cov_deriv_extrinsic_curvature(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ijj<DataType, SpatialDim, Frame> (*)(
          const tnsr::ii<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::Ijj<DataType, SpatialDim, Frame>&,
          const tnsr::AA<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::ijaa<DataType, SpatialDim, Frame>&)>(
          &::gh::covariant_deriv_of_extrinsic_curvature<DataType, SpatialDim,
                                                        Frame>),
      "GeneralRelativity.ComputeGhQuantities",
      "covariant_deriv_extrinsic_curvture", {{{-1., 1.}}}, used_for_size);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void test_spatial_ricci_tensor(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ii<DataType, SpatialDim, Frame> (*)(
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::ijaa<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&)>(
          &::gh::spatial_ricci_tensor<DataType, SpatialDim, Frame>),
      "GeneralRelativity.ComputeGhQuantities", "gh_spatial_ricci_tensor",
      {{{std::numeric_limits<double>::denorm_min(), 1.}}}, used_for_size);
}

// Test gh::ricci_tensor by comparing to specific values
// c.f. SpEC
void test_spatial_ricci_tensor_spec(const size_t grid_size_each_dimension,
                                    const std::array<double, 3>& lower_bound,
                                    const std::array<double, 3>& upper_bound) {
  using frame = Frame::Inertial;
  constexpr size_t VolumeDim = 3;
  // Setup grid
  Mesh<VolumeDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });

  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  const Direction<VolumeDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();
  const auto inertial_coords = [&slice_grid_points, &lower_bound]() {
    tnsr::I<DataVector, VolumeDim, frame> tmp(slice_grid_points, 0.);
    // +y direction
    get<1>(tmp) = 0.5;
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = 0; j < VolumeDim; ++j) {
        get<0>(tmp)[i * VolumeDim + j] =
            lower_bound[0] + 0.5 * static_cast<double>(i);
        get<2>(tmp)[i * VolumeDim + j] =
            lower_bound[2] + 0.5 * static_cast<double>(j);
      }
    }
    return tmp;
  }();

  auto local_inverse_spatial_metric =
      make_with_value<tnsr::II<DataVector, VolumeDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_phi =
      make_with_value<tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_d_pi =
      make_with_value<tnsr::iaa<DataVector, VolumeDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_d_phi =
      make_with_value<tnsr::ijaa<DataVector, VolumeDim, Frame::Inertial>>(
          inertial_coords, 0.);

  // Setting inverse_spatial_metric with values chosen to reproduce a result
  // from SpEC
  for (size_t i = 0; i < get<0>(inertial_coords).size(); ++i) {
    for (size_t j = 0; j < VolumeDim; ++j) {
      local_inverse_spatial_metric.get(0, j)[i] = 41.;
      local_inverse_spatial_metric.get(1, j)[i] = 43.;
      local_inverse_spatial_metric.get(2, j)[i] = 47.;
    }
  }
  // Setting pi AND phi with values chosen to reproduce a result from SpEC
  for (size_t i = 0; i < get<0>(inertial_coords).size(); ++i) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = 0; b <= VolumeDim; ++b) {
        // local_pi.get(a, b)[i] = 1.;
        local_phi.get(0, a, b)[i] = 3.;
        local_phi.get(1, a, b)[i] = 5.;
        local_phi.get(2, a, b)[i] = 7.;
      }
    }
  }
  // Setting local_d_phi with values chosen to reproduce a result from SpEC
  for (size_t i = 0; i < get<0>(inertial_coords).size(); ++i) {
    for (size_t a = 0; a <= VolumeDim; ++a) {
      for (size_t b = 0; b <= VolumeDim; ++b) {
        local_d_pi.get(0, a, b)[i] = 1.;
        local_d_phi.get(0, 0, a, b)[i] = 3.;
        local_d_phi.get(0, 1, a, b)[i] = 5.;
        local_d_phi.get(0, 2, a, b)[i] = 7.;
        local_d_pi.get(1, a, b)[i] = 53.;
        local_d_phi.get(1, 0, a, b)[i] = 59.;
        local_d_phi.get(1, 1, a, b)[i] = 61.;
        local_d_phi.get(1, 2, a, b)[i] = 67.;
        local_d_pi.get(2, a, b)[i] = 71.;
        local_d_phi.get(2, 0, a, b)[i] = 73.;
        local_d_phi.get(2, 1, a, b)[i] = 79.;
        local_d_phi.get(2, 2, a, b)[i] = 83.;
      }
    }
  }

  // Call tested function
  auto local_ricci_3 = gh::spatial_ricci_tensor(local_phi, local_d_phi,
                                                local_inverse_spatial_metric);

  // Initialize with values from SpEC
  auto spec_ricci_3 = local_ricci_3;
  get<0, 0>(spec_ricci_3) = 153230.;
  get<0, 1>(spec_ricci_3) = 5283.;
  get<0, 2>(spec_ricci_3) = -142541.;
  get<1, 0>(spec_ricci_3) = 5283.;
  get<1, 1>(spec_ricci_3) = 2497.;
  get<1, 2>(spec_ricci_3) = -928.;
  get<2, 2>(spec_ricci_3) = 141189.;

  // Compare with values from SpEC
  CHECK_ITERABLE_APPROX(local_ricci_3, spec_ricci_3);
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

  test_spatial_ricci_tensor<DataVector, 1, Frame::Grid>(used_for_size);
  test_spatial_ricci_tensor<DataVector, 2, Frame::Grid>(used_for_size);
  test_spatial_ricci_tensor<DataVector, 3, Frame::Grid>(used_for_size);
  test_spatial_ricci_tensor<DataVector, 1, Frame::Inertial>(used_for_size);
  test_spatial_ricci_tensor<DataVector, 2, Frame::Inertial>(used_for_size);
  test_spatial_ricci_tensor<DataVector, 3, Frame::Inertial>(used_for_size);

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
  test_spatial_ricci_tensor_spec(grid_size, lower_bound, upper_bound);

  // Check that compute items work correctly in the DataBox
  // First, check that the names are correct
  TestHelpers::db::test_compute_tag<
      gh::Tags::TimeDerivSpatialMetricCompute<3, Frame::Inertial>>(
      "dt(SpatialMetric)");
  TestHelpers::db::test_compute_tag<
      gh::Tags::TimeDerivLapseCompute<3, Frame::Inertial>>("dt(Lapse)");
  TestHelpers::db::test_compute_tag<
      gh::Tags::TimeDerivShiftCompute<3, Frame::Inertial>>("dt(Shift)");
  TestHelpers::db::test_compute_tag<
      gh::Tags::DerivSpatialMetricCompute<3, Frame::Inertial>>(
      "deriv(SpatialMetric)");
  TestHelpers::db::test_compute_tag<
      gh::Tags::DerivLapseCompute<3, Frame::Inertial>>("deriv(Lapse)");
  TestHelpers::db::test_compute_tag<
      gh::Tags::DerivShiftCompute<3, Frame::Inertial>>("deriv(Shift)");
  TestHelpers::db::test_compute_tag<gh::Tags::PhiCompute<3, Frame::Inertial>>(
      "Phi");
  TestHelpers::db::test_compute_tag<gh::Tags::PiCompute<3, Frame::Inertial>>(
      "Pi");
  TestHelpers::db::test_compute_tag<
      gh::Tags::ExtrinsicCurvatureCompute<3, Frame::Inertial>>(
      "ExtrinsicCurvature");
  TestHelpers::db::test_compute_tag<
      gh::Tags::TraceExtrinsicCurvatureCompute<3, Frame::Inertial>>(
      "TraceExtrinsicCurvature");
  TestHelpers::db::test_compute_tag<
      gh::ConstraintDamping::Tags::ConstraintGamma0Compute<3, Frame::Inertial>>(
      "ConstraintGamma0");
  TestHelpers::db::test_compute_tag<
      gh::ConstraintDamping::Tags::ConstraintGamma1Compute<3, Frame::Inertial>>(
      "ConstraintGamma1");
  TestHelpers::db::test_compute_tag<
      gh::ConstraintDamping::Tags::ConstraintGamma2Compute<3, Frame::Inertial>>(
      "ConstraintGamma2");
  TestHelpers::db::test_compute_tag<
      gh::Tags::SpacetimeDerivGaugeHCompute<3, Frame::Inertial>>(
      "SpacetimeDerivGaugeH");
  TestHelpers::db::test_compute_tag<
      gh::Tags::GaugeHImplicitFrom3p1QuantitiesCompute<3, Frame::Inertial>>(
      "GaugeH");

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

  const auto expected_phi =
      gh::phi(lapse, expected_deriv_lapse, shift, expected_deriv_shift,
              spatial_metric, expected_deriv_spatial_metric);
  const auto expected_pi =
      gh::pi(lapse, expected_dt_lapse, shift, expected_dt_shift, spatial_metric,
             expected_dt_spatial_metric, expected_phi);
  const auto expected_extrinsic_curvature = gh::extrinsic_curvature(
      spacetime_normal_vector, expected_pi, expected_phi);
  const auto expected_trace_extrinsic_curvature =
      trace(expected_extrinsic_curvature, inverse_spatial_metric);

  const auto box = db::create<
      db::AddSimpleTags<gr::Tags::Lapse<DataVector>,
                        gr::Tags::SpacetimeNormalVector<DataVector, 3>,
                        gr::Tags::InverseSpacetimeMetric<DataVector, 3>,
                        gh::Tags::Phi<DataVector, 3>>,
      db::AddComputeTags<
          gh::Tags::DerivSpatialMetricCompute<3, Frame::Inertial>,
          gh::Tags::DerivLapseCompute<3, Frame::Inertial>,
          gh::Tags::DerivShiftCompute<3, Frame::Inertial>>>(
      lapse, spacetime_normal_vector, inverse_spacetime_metric, expected_phi);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(db::get<::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>,
                                       tmpl::size_t<3>, Frame::Inertial>>(box)),
      expected_deriv_spatial_metric);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(db::get<::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                       tmpl::size_t<3>, Frame::Inertial>>(box)),
      expected_deriv_lapse);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(db::get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>,
                                       tmpl::size_t<3>, Frame::Inertial>>(box)),
      expected_deriv_shift);

  const auto other_box = db::create<
      db::AddSimpleTags<
          gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
          gr::Tags::SpacetimeNormalVector<DataVector, 3>,
          gr::Tags::InverseSpacetimeMetric<DataVector, 3>,
          gr::Tags::InverseSpatialMetric<DataVector, 3>,
          gh::Tags::Phi<DataVector, 3>, gh::Tags::Pi<DataVector, 3>>,
      db::AddComputeTags<
          gh::Tags::TimeDerivSpatialMetricCompute<3, Frame::Inertial>,
          gh::Tags::TimeDerivLapseCompute<3, Frame::Inertial>,
          gh::Tags::TimeDerivShiftCompute<3, Frame::Inertial>>>(
      lapse, shift, spacetime_normal_vector, inverse_spacetime_metric,
      inverse_spatial_metric, expected_phi, expected_pi);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(db::get<::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(
          other_box)),
      expected_dt_spatial_metric);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(db::get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(other_box)),
      expected_dt_lapse);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(
          db::get<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>(other_box)),
      expected_dt_shift);

  const auto ghvars_box = db::create<
      db::AddSimpleTags<gr::Tags::SpatialMetric<DataVector, 3>,
                        gr::Tags::Lapse<DataVector>,
                        gr::Tags::Shift<DataVector, 3>,
                        gr::Tags::SpacetimeNormalVector<DataVector, 3>,
                        gr::Tags::InverseSpatialMetric<DataVector, 3>,
                        ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>,
                                      tmpl::size_t<3>, Frame::Inertial>,
                        ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                      tmpl::size_t<3>, Frame::Inertial>,
                        ::Tags::deriv<gr::Tags::Shift<DataVector, 3>,
                                      tmpl::size_t<3>, Frame::Inertial>,
                        ::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>,
                        ::Tags::dt<gr::Tags::Lapse<DataVector>>,
                        ::Tags::dt<gr::Tags::Shift<DataVector, 3>>>,
      db::AddComputeTags<
          gh::Tags::PhiCompute<3, Frame::Inertial>,
          gh::Tags::PiCompute<3, Frame::Inertial>,
          gh::Tags::ExtrinsicCurvatureCompute<3, Frame::Inertial>,
          gh::Tags::TraceExtrinsicCurvatureCompute<3, Frame::Inertial>>>(
      spatial_metric, lapse, shift, spacetime_normal_vector,
      inverse_spatial_metric, expected_deriv_spatial_metric,
      expected_deriv_lapse, expected_deriv_shift, expected_dt_spatial_metric,
      expected_dt_lapse, expected_dt_shift);

  CHECK(db::get<gh::Tags::Phi<DataVector, 3>>(ghvars_box) == expected_phi);
  CHECK(db::get<gh::Tags::Pi<DataVector, 3>>(ghvars_box) == expected_pi);
  CHECK(db::get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(ghvars_box) ==
        expected_extrinsic_curvature);
  CHECK(db::get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(ghvars_box) ==
        expected_trace_extrinsic_curvature);
}
