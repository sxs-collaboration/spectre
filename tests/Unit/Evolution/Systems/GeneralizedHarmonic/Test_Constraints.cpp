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

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/GaussianPlusConstant.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintGammas.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/GaugeSource.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Tags::deriv

namespace {
template <typename DataType, size_t SpatialDim, typename Frame>
void test_three_index_constraint(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iaa<DataType, SpatialDim, Frame> (*)(
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &gh::three_index_constraint<DataType, SpatialDim, Frame>),
      "numpy", "subtract", {{{-1.0, 1.0}}}, used_for_size);
}

// Test the return-by-value gauge constraint function using random values
template <typename DataType, size_t SpatialDim, typename Frame>
void test_gauge_constraint_random(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::AA<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &gh::gauge_constraint<DataType, SpatialDim, Frame>),
      "TestFunctions", "gauge_constraint", {{{-1.0, 1.0}}}, used_for_size);
}

// Test the return-by-reference gauge constraint by comparing to Kerr-Schild
template <typename Solution>
void test_gauge_constraint_analytic(const Solution& solution,
                                    const size_t grid_size_each_dimension,
                                    const std::array<double, 3>& lower_bound,
                                    const std::array<double, 3>& upper_bound) {
  // Check vs. time-independent analytic solution
  // Set up grid
  Mesh<3> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1.0, 1.0, lower_bound[0], upper_bound[0]},
              Affine{-1.0, 1.0, lower_bound[1], upper_bound[1]},
              Affine{-1.0, 1.0, lower_bound[2], upper_bound[2]},
          });

  // Set up coordinates
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
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<DataVector, 3>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients for computing the gauge constraint
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto gauge_function = gh::gauge_source(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, spatial_metric,
      trace(gr::extrinsic_curvature(lapse, shift, d_shift, spatial_metric,
                                    dt_spatial_metric, d_spatial_metric),
            inverse_spatial_metric),
      trace_last_indices(gr::christoffel_first_kind(d_spatial_metric),
                         inverse_spatial_metric));
  const auto phi =
      gh::phi(lapse, d_lapse, shift, d_shift, spatial_metric, d_spatial_metric);
  const auto pi = gh::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
                         dt_spatial_metric, phi);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<DataVector, 3, Frame::Inertial>(lapse);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);

  // Get the constraint, and check that it vanishes
  auto constraint =
      make_with_value<tnsr::a<DataVector, 3, Frame::Inertial>>(x, 0.0);
  const gsl::not_null<tnsr::a<DataVector, 3, Frame::Inertial>*>
      constraint_pointer = &(constraint);
  gh::gauge_constraint(constraint_pointer, gauge_function, normal_one_form,
                       normal_vector, inverse_spatial_metric,
                       inverse_spacetime_metric, pi, phi);
  CHECK_ITERABLE_APPROX(constraint,
                        make_with_value<decltype(constraint)>(x, 0.0));
}

// Test the return-by-value two-index constraint function using random values
template <typename DataType, size_t SpatialDim, typename Frame>
void test_two_index_constraint_random(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ia<DataType, SpatialDim, Frame> (*)(
          const tnsr::ab<DataType, SpatialDim, Frame>&,
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::AA<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::ijaa<DataType, SpatialDim, Frame>&,
          const Scalar<DataType>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &gh::two_index_constraint<DataType, SpatialDim, Frame>),
      "TestFunctions", "two_index_constraint", {{{-1.0, 1.0}}}, used_for_size);
}

// Test the return-by-reference two-index constraint
// by comparing to a time-independent analytic solution (e.g. Kerr-Schild)
template <typename Solution>
void test_two_index_constraint_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, const double error_tolerance) {
  // Shorter names for tags.
  using SpacetimeMetric = gr::Tags::SpacetimeMetric<DataVector, 3>;
  using Pi = ::gh::Tags::Pi<DataVector, 3>;
  using Phi = ::gh::Tags::Phi<DataVector, 3>;
  using GaugeH = ::gh::Tags::GaugeH<DataVector, 3>;
  using VariablesTags = tmpl::list<SpacetimeMetric, Pi, Phi, GaugeH>;

  // Check vs. time-independent analytic solution
  // Set up grid
  const size_t data_size = pow<3>(grid_size_each_dimension);
  Mesh<3> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1.0, 1.0, lower_bound[0], upper_bound[0]},
              Affine{-1.0, 1.0, lower_bound[1], upper_bound[1]},
              Affine{-1.0, 1.0, lower_bound[2], upper_bound[2]},
          });

  // Set up coordinates
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
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<DataVector, 3>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Compute quantities from variables for the two-index constraint
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<DataVector, 3, Frame::Inertial>(lapse);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);

  // Arbitrary choice for gamma2
  const auto gamma2 = make_with_value<Scalar<DataVector>>(x, 4.0);

  // Compute derivatives d_psi, d_phi, d_pi, and d_gauge_function numerically
  Variables<VariablesTags> gh_vars(data_size);
  auto& spacetime_metric = get<SpacetimeMetric>(gh_vars);
  auto& pi = get<Pi>(gh_vars);
  auto& phi = get<Phi>(gh_vars);
  auto& gauge_function = get<GaugeH>(gh_vars);
  spacetime_metric = gr::spacetime_metric(lapse, shift, spatial_metric);
  phi =
      gh::phi(lapse, d_lapse, shift, d_shift, spatial_metric, d_spatial_metric);
  pi = gh::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
              dt_spatial_metric, phi);
  gauge_function = gh::gauge_source(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, spatial_metric,
      trace(gr::extrinsic_curvature(lapse, shift, d_shift, spatial_metric,
                                    dt_spatial_metric, d_spatial_metric),
            inverse_spatial_metric),
      trace_last_indices(gr::christoffel_first_kind(d_spatial_metric),
                         inverse_spatial_metric));

  // Compute numerical derivatives of psi,pi,phi,H
  const auto gh_derivs =
      partial_derivatives<VariablesTags, VariablesTags, 3, Frame::Inertial>(
          gh_vars, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_spacetime_metric =
      get<Tags::deriv<SpacetimeMetric, tmpl::size_t<3>, Frame::Inertial>>(
          gh_derivs);
  const auto& d_pi =
      get<Tags::deriv<Pi, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);
  const auto& d_phi =
      get<Tags::deriv<Phi, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);
  const auto& d_gauge_function =
      get<Tags::deriv<GaugeH, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);

  // Populate spacetime derivative of gauge function
  const auto d4_gauge_function = [&x, &d_gauge_function]() {
    auto tmp =
        make_with_value<tnsr::ab<DataVector, 3, Frame::Inertial>>(x, 0.0);
    for (size_t a = 0; a < 4; ++a) {
      for (size_t i = 0; i < 3; ++i) {
        tmp.get(i + 1, a) = d_gauge_function.get(i, a);
      }
      // Populate the time derivative with NaN as a check that those are not
      // being used in calculating constraints
      tmp.get(0, a) = std::numeric_limits<double>::signaling_NaN();
    }
    return tmp;
  }();

  // Compute the three-index constraint
  const auto three_index_constraint =
      gh::three_index_constraint(d_spacetime_metric, phi);

  // Get the constraint, and check that it vanishes to error_tolerance
  auto two_index_constraint =
      make_with_value<tnsr::ia<DataVector, 3, Frame::Inertial>>(
          x, std::numeric_limits<double>::signaling_NaN());
  gh::two_index_constraint(make_not_null(&two_index_constraint),
                           d4_gauge_function, normal_one_form, normal_vector,
                           inverse_spatial_metric, inverse_spacetime_metric, pi,
                           phi, d_pi, d_phi, gamma2, three_index_constraint);

  Approx numerical_approx =
      Approx::custom().epsilon(error_tolerance).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      two_index_constraint,
      make_with_value<decltype(two_index_constraint)>(x, 0.0),
      numerical_approx);
}

// Test the return-by-value four-index constraint function using random values
template <typename DataType, size_t SpatialDim, typename Frame>
void test_four_index_constraint_random(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iaa<DataType, SpatialDim, Frame> (*)(
          const tnsr::ijaa<DataType, SpatialDim, Frame>&)>(
          &gh::four_index_constraint<DataType, SpatialDim, Frame>),
      "TestFunctions", "four_index_constraint", {{{-1.0, 1.0}}}, used_for_size);
}

// Test the return-by-reference four-index constraint
// by comparing to a time-independent analytic solution (e.g. Kerr-Schild)
template <typename Solution>
void test_four_index_constraint_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound, const double error_tolerance) {
  // Shorter names for tags.
  using Phi = ::gh::Tags::Phi<DataVector, 3>;
  using VariablesTags = tmpl::list<Phi>;

  // Check vs. time-independent analytic solution
  // Set up grid
  const size_t data_size = pow<3>(grid_size_each_dimension);
  Mesh<3> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1.0, 1.0, lower_bound[0], upper_bound[0]},
              Affine{-1.0, 1.0, lower_bound[1], upper_bound[1]},
              Affine{-1.0, 1.0, lower_bound[2], upper_bound[2]},
          });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Compute derivative d_phi numerically
  Variables<VariablesTags> gh_vars(data_size);
  get<Phi>(gh_vars) =
      gh::phi(lapse, d_lapse, shift, d_shift, spatial_metric, d_spatial_metric);

  // Compute numerical derivatives of phi
  const auto gh_derivs =
      partial_derivatives<VariablesTags, VariablesTags, 3, Frame::Inertial>(
          gh_vars, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_phi =
      get<Tags::deriv<Phi, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);

  // Get the constraint, and check that it vanishes to error_tolerance
  auto four_index_constraint =
      make_with_value<tnsr::iaa<DataVector, 3, Frame::Inertial>>(
          x, std::numeric_limits<double>::signaling_NaN());
  gh::four_index_constraint(make_not_null(&four_index_constraint), d_phi);

  Approx numerical_approx =
      Approx::custom().epsilon(error_tolerance).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      four_index_constraint,
      make_with_value<decltype(four_index_constraint)>(x, 0.0),
      numerical_approx);
}

// Test the return-by-value F constraint function using random values
template <typename DataType, size_t SpatialDim, typename Frame>
void test_f_constraint_random(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::ab<DataType, SpatialDim, Frame>&,
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::AA<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::ijaa<DataType, SpatialDim, Frame>&,
          const Scalar<DataType>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &gh::f_constraint<DataType, SpatialDim, Frame>),
      "TestFunctions", "f_constraint", {{{-1.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::ab<DataType, SpatialDim, Frame>&,
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::AA<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::ijaa<DataType, SpatialDim, Frame>&,
          const Scalar<DataType>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&)>(
          &gh::f_constraint<DataType, SpatialDim, Frame>),
      "TestFunctions", "f_constraint_with_stress_energy", {{{-1.0, 1.0}}},
      used_for_size);
}

// Test the return-by-reference F constraint
// by comparing to a time-independent analytic solution (e.g. Kerr-Schild)
template <typename Solution>
void test_f_constraint_analytic(const Solution& solution,
                                const size_t grid_size_each_dimension,
                                const std::array<double, 3>& lower_bound,
                                const std::array<double, 3>& upper_bound,
                                const double error_tolerance) {
  // Shorter names for tags.
  using SpacetimeMetric = gr::Tags::SpacetimeMetric<DataVector, 3>;
  using Pi = ::gh::Tags::Pi<DataVector, 3>;
  using Phi = ::gh::Tags::Phi<DataVector, 3>;
  using GaugeH = ::gh::Tags::GaugeH<DataVector, 3>;
  using VariablesTags = tmpl::list<SpacetimeMetric, Pi, Phi, GaugeH>;

  // Check vs. time-independent analytic solution
  // Set up grid
  const size_t data_size = pow<3>(grid_size_each_dimension);
  Mesh<3> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1.0, 1.0, lower_bound[0], upper_bound[0]},
              Affine{-1.0, 1.0, lower_bound[1], upper_bound[1]},
              Affine{-1.0, 1.0, lower_bound[2], upper_bound[2]},
          });

  // Set up coordinates
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
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<DataVector, 3>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Compute quantities from variables for the two-index constraint
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<DataVector, 3, Frame::Inertial>(lapse);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);

  // Arbitrary choice for gamma2
  const auto gamma2 = make_with_value<Scalar<DataVector>>(x, 4.0);

  // Compute derivatives d_psi, d_phi, d_pi, and d_gauge_function numerically
  Variables<VariablesTags> gh_vars(data_size);
  auto& spacetime_metric = get<SpacetimeMetric>(gh_vars);
  auto& pi = get<Pi>(gh_vars);
  auto& phi = get<Phi>(gh_vars);
  auto& gauge_function = get<GaugeH>(gh_vars);
  spacetime_metric = gr::spacetime_metric(lapse, shift, spatial_metric);
  phi =
      gh::phi(lapse, d_lapse, shift, d_shift, spatial_metric, d_spatial_metric);
  pi = gh::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
              dt_spatial_metric, phi);
  gauge_function = gh::gauge_source(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, spatial_metric,
      trace(gr::extrinsic_curvature(lapse, shift, d_shift, spatial_metric,
                                    dt_spatial_metric, d_spatial_metric),
            inverse_spatial_metric),
      trace_last_indices(gr::christoffel_first_kind(d_spatial_metric),
                         inverse_spatial_metric));

  // Compute numerical derivatives of psi,pi,phi,H
  const auto gh_derivs =
      partial_derivatives<VariablesTags, VariablesTags, 3, Frame::Inertial>(
          gh_vars, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_spacetime_metric =
      get<Tags::deriv<SpacetimeMetric, tmpl::size_t<3>, Frame::Inertial>>(
          gh_derivs);
  const auto& d_pi =
      get<Tags::deriv<Pi, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);
  const auto& d_phi =
      get<Tags::deriv<Phi, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);
  const auto& d_gauge_function =
      get<Tags::deriv<GaugeH, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);

  // Populate spacetime derivative of gauge function
  const auto d4_gauge_function = [&x, &d_gauge_function]() {
    auto tmp =
        make_with_value<tnsr::ab<DataVector, 3, Frame::Inertial>>(x, 0.0);
    for (size_t a = 0; a < 4; ++a) {
      for (size_t i = 0; i < 3; ++i) {
        tmp.get(i + 1, a) = d_gauge_function.get(i, a);
      }
      // Populate the time derivative with NaN as a check that those are not
      // being used in calculating constraints
      tmp.get(0, a) = std::numeric_limits<double>::signaling_NaN();
    }
    return tmp;
  }();

  // Compute the three-index constraint
  const auto three_index_constraint =
      gh::three_index_constraint(d_spacetime_metric, phi);

  // Get the constraint, and check that it vanishes to error_tolerance
  auto f_constraint = make_with_value<tnsr::a<DataVector, 3, Frame::Inertial>>(
      x, std::numeric_limits<double>::signaling_NaN());
  gh::f_constraint(make_not_null(&f_constraint), gauge_function,
                   d4_gauge_function, normal_one_form, normal_vector,
                   inverse_spatial_metric, inverse_spacetime_metric, pi, phi,
                   d_pi, d_phi, gamma2, three_index_constraint);

  Approx numerical_approx =
      Approx::custom().epsilon(error_tolerance).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(f_constraint,
                               make_with_value<decltype(f_constraint)>(x, 0.0),
                               numerical_approx);
}

// Test the return-by-value constraint energy function using random values
template <typename DataType, size_t SpatialDim, typename Frame>
void test_constraint_energy_random(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::ia<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&, const Scalar<DataType>&,
          double, double, double, double)>(
          &gh::constraint_energy<DataType, SpatialDim, Frame>),
      "TestFunctions", "constraint_energy", {{{-1.0, 1.0}}}, used_for_size);
}

// Test the return-by-reference constraint energy
// by comparing to a time-independent analytic solution (e.g. Kerr-Schild)
template <typename Solution>
void test_constraint_energy_analytic(const Solution& solution,
                                     const size_t grid_size_each_dimension,
                                     const std::array<double, 3>& lower_bound,
                                     const std::array<double, 3>& upper_bound,
                                     const double error_tolerance) {
  // Shorter names for tags.
  using SpacetimeMetric = gr::Tags::SpacetimeMetric<DataVector, 3>;
  using Pi = ::gh::Tags::Pi<DataVector, 3>;
  using Phi = ::gh::Tags::Phi<DataVector, 3>;
  using GaugeH = ::gh::Tags::GaugeH<DataVector, 3>;
  using VariablesTags = tmpl::list<SpacetimeMetric, Pi, Phi, GaugeH>;

  // Check vs. time-independent analytic solution
  // Set up grid
  const size_t data_size = pow<3>(grid_size_each_dimension);
  Mesh<3> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1.0, 1.0, lower_bound[0], upper_bound[0]},
              Affine{-1.0, 1.0, lower_bound[1], upper_bound[1]},
              Affine{-1.0, 1.0, lower_bound[2], upper_bound[2]},
          });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataVector>>>(vars);
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<DataVector, 3>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(vars);

  const auto determinant_and_inverse_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& determinant_spatial_metric =
      determinant_and_inverse_spatial_metric.first;
  const auto& inverse_spatial_metric =
      determinant_and_inverse_spatial_metric.second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<DataVector, 3, Frame::Inertial>(lapse);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);

  // Compute derivative d_phi numerically
  Variables<VariablesTags> gh_vars(data_size);
  auto& spacetime_metric = get<SpacetimeMetric>(gh_vars);
  auto& pi = get<Pi>(gh_vars);
  auto& phi = get<Phi>(gh_vars);
  auto& gauge_function = get<GaugeH>(gh_vars);
  spacetime_metric = gr::spacetime_metric(lapse, shift, spatial_metric);
  phi =
      gh::phi(lapse, d_lapse, shift, d_shift, spatial_metric, d_spatial_metric);
  pi = gh::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
              dt_spatial_metric, phi);
  gauge_function = gh::gauge_source(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, spatial_metric,
      trace(gr::extrinsic_curvature(lapse, shift, d_shift, spatial_metric,
                                    dt_spatial_metric, d_spatial_metric),
            inverse_spatial_metric),
      trace_last_indices(gr::christoffel_first_kind(d_spatial_metric),
                         inverse_spatial_metric));

  // Compute numerical derivatives of psi,pi,phi,H
  const auto gh_derivs =
      partial_derivatives<VariablesTags, VariablesTags, 3, Frame::Inertial>(
          gh_vars, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_spacetime_metric =
      get<Tags::deriv<SpacetimeMetric, tmpl::size_t<3>, Frame::Inertial>>(
          gh_derivs);
  const auto& d_pi =
      get<Tags::deriv<Pi, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);
  const auto& d_phi =
      get<Tags::deriv<Phi, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);
  const auto& d_gauge_function =
      get<Tags::deriv<GaugeH, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);

  // Populate spacetime derivative of gauge function
  const auto d4_gauge_function = [&x, &d_gauge_function]() {
    auto tmp =
        make_with_value<tnsr::ab<DataVector, 3, Frame::Inertial>>(x, 0.0);
    for (size_t a = 0; a < 4; ++a) {
      for (size_t i = 0; i < 3; ++i) {
        tmp.get(i + 1, a) = d_gauge_function.get(i, a);
      }
      // Populate the time derivative with NaN as a check that those are not
      // being used in calculating constraints
      tmp.get(0, a) = std::numeric_limits<double>::signaling_NaN();
    }
    return tmp;
  }();

  // Arbitrary choice for gamma2
  const auto gamma2 = make_with_value<Scalar<DataVector>>(x, 4.0);

  // Get the constraints
  const auto gauge_constraint = gh::gauge_constraint(
      gauge_function, normal_one_form, normal_vector, inverse_spatial_metric,
      inverse_spacetime_metric, pi, phi);
  const auto three_index_constraint =
      gh::three_index_constraint(d_spacetime_metric, phi);
  const auto four_index_constraint = gh::four_index_constraint(d_phi);
  const auto two_index_constraint = gh::two_index_constraint(
      d4_gauge_function, normal_one_form, normal_vector, inverse_spatial_metric,
      inverse_spacetime_metric, pi, phi, d_pi, d_phi, gamma2,
      three_index_constraint);
  const auto f_constraint = gh::f_constraint(
      gauge_function, d4_gauge_function, normal_one_form, normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi, d_pi, d_phi,
      gamma2, three_index_constraint);

  auto constraint_energy = make_with_value<Scalar<DataVector>>(
      x, std::numeric_limits<double>::signaling_NaN());
  gh::constraint_energy(
      make_not_null(&constraint_energy), gauge_constraint, f_constraint,
      two_index_constraint, three_index_constraint, four_index_constraint,
      inverse_spatial_metric, determinant_spatial_metric, 2.0, -3.0, 4.0, -5.0);

  Approx numerical_approx =
      Approx::custom().epsilon(error_tolerance).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      constraint_energy, make_with_value<decltype(constraint_energy)>(x, 0.0),
      numerical_approx);
}

// Test compute items for various constraints via insertion and retrieval
// in a databox
template <typename Solution>
void test_constraint_compute_items(const Solution& solution,
                                   const size_t grid_size,
                                   const std::array<double, 3>& lower_bound,
                                   const std::array<double, 3>& upper_bound) {
  // Check that compute items are named correctly
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
      gh::Tags::GaugeHImplicitFrom3p1QuantitiesCompute<3, Frame::Inertial>>(
      "GaugeH");
  TestHelpers::db::test_compute_tag<
      gh::Tags::SpacetimeDerivGaugeHCompute<3, Frame::Inertial>>(
      "SpacetimeDerivGaugeH");
  TestHelpers::db::test_compute_tag<
      gh::Tags::GaugeConstraintCompute<3, Frame::Inertial>>("GaugeConstraint");
  TestHelpers::db::test_compute_tag<
      gh::Tags::FConstraintCompute<3, Frame::Inertial>>("FConstraint");
  TestHelpers::db::test_compute_tag<
      gh::Tags::TwoIndexConstraintCompute<3, Frame::Inertial>>(
      "TwoIndexConstraint");
  TestHelpers::db::test_compute_tag<
      gh::Tags::ThreeIndexConstraintCompute<3, Frame::Inertial>>(
      "ThreeIndexConstraint");
  TestHelpers::db::test_compute_tag<
      gh::Tags::FourIndexConstraintCompute<3, Frame::Inertial>>(
      "FourIndexConstraint");
  TestHelpers::db::test_compute_tag<
      gh::Tags::ConstraintEnergyCompute<3, Frame::Inertial>>(
      "ConstraintEnergy");

  // Check vs. time-independent analytic solution
  // Set up grid
  const size_t data_size = pow<3>(grid_size);
  Mesh<3> mesh{grid_size, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{Affine{-1.0, 1.0, lower_bound[0], upper_bound[0]},
                   Affine{-1.0, 1.0, lower_bound[1], upper_bound[1]},
                   Affine{-1.0, 1.0, lower_bound[2], upper_bound[2]}});

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(vars);
  const auto& time_deriv_lapse =
      get<Tags::dt<gr::Tags::Lapse<DataVector>>>(vars);
  const auto& deriv_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(vars);
  const auto& deriv_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& time_deriv_shift =
      get<Tags::dt<gr::Tags::Shift<DataVector, 3>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(vars);
  const auto& time_deriv_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(vars);
  const auto& deriv_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Compute quantities from variables for the two-index constraint
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);
  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<DataVector, 3, Frame::Inertial>(lapse);
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& det_spatial_metric = det_and_inv.first;
  const auto& inverse_spatial_metric = det_and_inv.second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);

  const auto christoffel_first_kind =
      gr::christoffel_first_kind(deriv_spatial_metric);
  const auto trace_christoffel_first_kind =
      trace_last_indices(christoffel_first_kind, inverse_spatial_metric);

  // Compute derivatives d_spacetime_metric, d_phi, d_pi, and d_gauge_function
  // numerically
  // First, prepare
  using VariablesTags =
      tmpl::list<gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
                 gh::Tags::GaugeH<DataVector, 3>>;

  Variables<VariablesTags> gh_vars(data_size);
  auto& pi = get<gh::Tags::Pi<DataVector, 3>>(gh_vars);
  auto& phi = get<gh::Tags::Phi<DataVector, 3>>(gh_vars);
  auto& gauge_source = get<gh::Tags::GaugeH<DataVector, 3>>(gh_vars);
  phi = gh::phi(lapse, deriv_lapse, shift, deriv_shift, spatial_metric,
                deriv_spatial_metric);
  pi = gh::pi(lapse, time_deriv_lapse, shift, time_deriv_shift, spatial_metric,
              time_deriv_spatial_metric, phi);
  const auto extrinsic_curvature =
      gh::extrinsic_curvature(spacetime_normal_vector, pi, phi);
  const auto trace_extrinsic_curvature =
      trace(extrinsic_curvature, inverse_spatial_metric);
  gauge_source =
      gh::gauge_source(lapse, time_deriv_lapse, deriv_lapse, shift,
                       time_deriv_shift, deriv_shift, spatial_metric,
                       trace_extrinsic_curvature, trace_christoffel_first_kind);
  // Second, compute derivatives
  const auto gh_derivs =
      partial_derivatives<VariablesTags, VariablesTags, 3, Frame::Inertial>(
          gh_vars, mesh, coord_map.inv_jacobian(x_logical));
  const auto& deriv_pi =
      get<Tags::deriv<gh::Tags::Pi<DataVector, 3>, tmpl::size_t<3>,
                      Frame::Inertial>>(gh_derivs);
  const auto& deriv_phi =
      get<Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                      Frame::Inertial>>(gh_derivs);
  const auto& deriv_gauge_source =
      get<Tags::deriv<gh::Tags::GaugeH<DataVector, 3>, tmpl::size_t<3>,
                      Frame::Inertial>>(gh_derivs);

  // The spatial derivative of the spacetime metric is needed to evaluate
  // the three-index constraint. Here, we choose random values for this tensor,
  // so that the constraint is deliberately nonzero.
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);
  const auto& deriv_spacetime_metric =
      make_with_random_values<tnsr::iaa<DataVector, 3, Frame::Inertial>>(
          nn_generator, nn_dist, get<0, 0>(pi));

  auto time_deriv_gauge_source =
      make_with_value<tnsr::a<DataVector, 3, Frame::Inertial>>(x, 0.);
  get<0>(time_deriv_gauge_source) = 0.05;
  get<1>(time_deriv_gauge_source) = 0.06;
  get<2>(time_deriv_gauge_source) = 0.07;
  get<3>(time_deriv_gauge_source) = -0.05;

  tnsr::ab<DataVector, 3, Frame::Inertial> spacetime_deriv_gauge_source{};
  gh::Tags::SpacetimeDerivGaugeHCompute<3, Frame::Inertial>::function(
      make_not_null(&spacetime_deriv_gauge_source), time_deriv_gauge_source,
      deriv_gauge_source);

  // Make DampingFunctions for the constraint damping parameters
  // Note: these parameters are taken from SpEC single-black-hole simulations
  constexpr double constant_02 = 0.001;
  constexpr double amplitude_0 = 3.0;
  constexpr double amplitude_2 = 1.0;
  constexpr double constant_1 = -1.0;
  constexpr double amplitude_1 = 0.0;
  constexpr double width = 11.3137084989848;  // sqrt(128.0)
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};

  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  // Insert into databox
  const auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::FunctionsOfTimeInitialize, ::Tags::Time,
          domain::Tags::Coordinates<3, Frame::Inertial>,
          gr::Tags::SpatialMetric<DataVector, 3>, gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<DataVector, 3>,
          ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>,
          ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>,
          ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>,
          ::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>,
          ::Tags::dt<gr::Tags::Lapse<DataVector>>,
          ::Tags::dt<gr::Tags::Shift<DataVector, 3>>,
          ::Tags::deriv<gh::Tags::GaugeH<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>,
          ::Tags::dt<gh::Tags::GaugeH<DataVector, 3>>,
          ::Tags::deriv<gr::Tags::SpacetimeMetric<DataVector, 3>,
                        tmpl::size_t<3>, Frame::Inertial>,
          ::Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>,
          ::Tags::deriv<gh::Tags::Pi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>,
          gh::ConstraintDamping::Tags::DampingFunctionGamma0<3,
                                                             Frame::Inertial>,
          gh::ConstraintDamping::Tags::DampingFunctionGamma1<3,
                                                             Frame::Inertial>,
          gh::ConstraintDamping::Tags::DampingFunctionGamma2<3,
                                                             Frame::Inertial>>,
      db::AddComputeTags<
          gh::ConstraintDamping::Tags::ConstraintGamma0Compute<3,
                                                               Frame::Inertial>,
          gh::ConstraintDamping::Tags::ConstraintGamma1Compute<3,
                                                               Frame::Inertial>,
          gh::ConstraintDamping::Tags::ConstraintGamma2Compute<3,
                                                               Frame::Inertial>,
          gr::Tags::SpacetimeNormalOneFormCompute<DataVector, 3,
                                                  Frame::Inertial>,
          gr::Tags::SpacetimeNormalVectorCompute<DataVector, 3,
                                                 Frame::Inertial>,
          gr::Tags::DetAndInverseSpatialMetricCompute<DataVector, 3,
                                                      Frame::Inertial>,
          gr::Tags::InverseSpacetimeMetricCompute<DataVector, 3,
                                                  Frame::Inertial>,
          gr::Tags::SpatialChristoffelFirstKindCompute<DataVector, 3,
                                                       Frame::Inertial>,
          gr::Tags::TraceSpatialChristoffelFirstKindCompute<DataVector, 3,
                                                            Frame::Inertial>,
          gh::Tags::PhiCompute<3, Frame::Inertial>,
          gh::Tags::PiCompute<3, Frame::Inertial>,
          gh::Tags::ExtrinsicCurvatureCompute<3, Frame::Inertial>,
          gh::Tags::TraceExtrinsicCurvatureCompute<3, Frame::Inertial>,
          gh::Tags::GaugeHImplicitFrom3p1QuantitiesCompute<3, Frame::Inertial>,
          gh::Tags::SpacetimeDerivGaugeHCompute<3, Frame::Inertial>,
          gh::Tags::FourIndexConstraintCompute<3, Frame::Inertial>,
          gh::Tags::ThreeIndexConstraintCompute<3, Frame::Inertial>,
          gh::Tags::TwoIndexConstraintCompute<3, Frame::Inertial>,
          gh::Tags::GaugeConstraintCompute<3, Frame::Inertial>,
          gh::Tags::FConstraintCompute<3, Frame::Inertial>,
          gh::Tags::ConstraintEnergyCompute<3, Frame::Inertial>>>(
      std::move(functions_of_time),
      std::numeric_limits<double>::signaling_NaN(), x,

      spatial_metric, lapse, shift, deriv_spatial_metric, deriv_lapse,
      deriv_shift, time_deriv_spatial_metric, time_deriv_lapse,
      time_deriv_shift, deriv_gauge_source, time_deriv_gauge_source,
      deriv_spacetime_metric, deriv_phi, deriv_pi,
      std::move(
          static_cast<std::unique_ptr<
              gh::ConstraintDamping::DampingFunction<3, Frame::Inertial>>>(
              std::make_unique<gh::ConstraintDamping::GaussianPlusConstant<
                  3, Frame::Inertial>>(constant_02, amplitude_0, width,
                                       center))),
      std::move(
          static_cast<std::unique_ptr<
              gh::ConstraintDamping::DampingFunction<3, Frame::Inertial>>>(
              std::make_unique<gh::ConstraintDamping::GaussianPlusConstant<
                  3, Frame::Inertial>>(constant_1, amplitude_1, width,
                                       center))),
      std::move(
          static_cast<std::unique_ptr<
              gh::ConstraintDamping::DampingFunction<3, Frame::Inertial>>>(
              std::make_unique<gh::ConstraintDamping::GaussianPlusConstant<
                  3, Frame::Inertial>>(constant_02, amplitude_2, width,
                                       center))));

  // Compute tested quantities locally
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      empty_functions_of_time{};
  Scalar<DataVector> gamma0{};
  gh::ConstraintDamping::Tags::ConstraintGamma0Compute<3, Frame::Inertial>::
      function(make_not_null(&gamma0),
               gh::ConstraintDamping::GaussianPlusConstant<3, Frame::Inertial>{
                   constant_02, amplitude_0, width, center},
               x, std::numeric_limits<double>::signaling_NaN(),
               empty_functions_of_time);
  Scalar<DataVector> gamma1{};
  gh::ConstraintDamping::Tags::ConstraintGamma1Compute<3, Frame::Inertial>::
      function(make_not_null(&gamma1),
               gh::ConstraintDamping::GaussianPlusConstant<3, Frame::Inertial>{
                   constant_1, amplitude_1, width, center},
               x, std::numeric_limits<double>::signaling_NaN(),
               empty_functions_of_time);
  Scalar<DataVector> gamma2{};
  gh::ConstraintDamping::Tags::ConstraintGamma2Compute<3, Frame::Inertial>::
      function(make_not_null(&gamma2),
               gh::ConstraintDamping::GaussianPlusConstant<3, Frame::Inertial>{
                   constant_02, amplitude_2, width, center},
               x, std::numeric_limits<double>::signaling_NaN(),
               empty_functions_of_time);

  const auto four_index_constraint = gh::four_index_constraint(deriv_phi);
  const auto three_index_constraint =
      gh::three_index_constraint(deriv_spacetime_metric, phi);
  const auto two_index_constraint = gh::two_index_constraint(
      spacetime_deriv_gauge_source, spacetime_normal_one_form,
      spacetime_normal_vector, inverse_spatial_metric, inverse_spacetime_metric,
      pi, phi, deriv_pi, deriv_phi, gamma2, three_index_constraint);
  const auto gauge_constraint = gh::gauge_constraint(
      gauge_source, spacetime_normal_one_form, spacetime_normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi);
  const auto f_constraint = gh::f_constraint(
      gauge_source, spacetime_deriv_gauge_source, spacetime_normal_one_form,
      spacetime_normal_vector, inverse_spatial_metric, inverse_spacetime_metric,
      pi, phi, deriv_pi, deriv_phi, gamma2, three_index_constraint);
  const auto constraint_energy = gh::constraint_energy(
      gauge_constraint, f_constraint, two_index_constraint,
      three_index_constraint, four_index_constraint, inverse_spatial_metric,
      det_spatial_metric);

  // Check that their compute items in databox furnish identical values
  CHECK(db::get<gh::ConstraintDamping::Tags::ConstraintGamma0>(box) == gamma0);
  CHECK(db::get<gh::ConstraintDamping::Tags::ConstraintGamma1>(box) == gamma1);
  CHECK(db::get<gh::ConstraintDamping::Tags::ConstraintGamma2>(box) == gamma2);
  CHECK(db::get<gh::Tags::GaugeH<DataVector, 3>>(box) == gauge_source);
  CHECK(db::get<gh::Tags::SpacetimeDerivGaugeH<DataVector, 3>>(box) ==
        spacetime_deriv_gauge_source);
  CHECK(db::get<gh::Tags::FourIndexConstraint<DataVector, 3>>(box) ==
        four_index_constraint);
  CHECK(db::get<gh::Tags::ThreeIndexConstraint<DataVector, 3>>(box) ==
        three_index_constraint);
  CHECK(db::get<gh::Tags::TwoIndexConstraint<DataVector, 3>>(box) ==
        two_index_constraint);
  CHECK(db::get<gh::Tags::GaugeConstraint<DataVector, 3>>(box) ==
        gauge_constraint);
  CHECK(db::get<gh::Tags::FConstraint<DataVector, 3>>(box) == f_constraint);
  CHECK(db::get<gh::Tags::ConstraintEnergy<DataVector, 3>>(box) ==
        constraint_energy);
}

void three_index_constraint() {
  test_three_index_constraint<DataVector, 1, Frame::Grid>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<DataVector, 1, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<double, 1, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<double, 1, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());

  test_three_index_constraint<DataVector, 2, Frame::Grid>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<DataVector, 2, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<double, 2, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<double, 2, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());

  test_three_index_constraint<DataVector, 3, Frame::Grid>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<DataVector, 3, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<double, 3, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<double, 3, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());
}

void gauge_constraint() {
  // Test the gauge constraint against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 3;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};

  test_gauge_constraint_analytic(solution, grid_size, lower_bound, upper_bound);

  // Test the gauge constraint with random numbers
  test_gauge_constraint_random<DataVector, 1, Frame::Grid>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<DataVector, 1, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<double, 1, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_gauge_constraint_random<double, 1, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());

  test_gauge_constraint_random<DataVector, 2, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<double, 2, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_gauge_constraint_random<double, 2, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());

  test_gauge_constraint_random<DataVector, 3, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<double, 3, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_gauge_constraint_random<double, 3, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());
}

void two_index_constraint() {
  // Test the two-index constraint against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> dimensionless_spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, dimensionless_spin, center);

  const size_t grid_size = 5;
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};

  // Note: looser numerical tolerance because this check
  // uses numerical derivatives
  test_two_index_constraint_analytic(
      solution, grid_size, lower_bound, upper_bound,
      std::numeric_limits<double>::epsilon() * 1.e8);

  // Test the two-index constraint with random numbers
  test_two_index_constraint_random<DataVector, 1, Frame::Grid>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_two_index_constraint_random<DataVector, 1, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_two_index_constraint_random<double, 1, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_two_index_constraint_random<double, 1, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());

  test_two_index_constraint_random<DataVector, 2, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_two_index_constraint_random<double, 2, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_two_index_constraint_random<double, 2, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());

  test_two_index_constraint_random<DataVector, 3, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_two_index_constraint_random<double, 3, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_two_index_constraint_random<double, 3, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());
}

void four_index_constraint() {
  // Test the four-index constraint against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> dimensionless_spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, dimensionless_spin, center);

  const size_t grid_size = 5;
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};

  // Note: looser numerical tolerance because this check
  // uses numerical derivatives
  test_four_index_constraint_analytic(
      solution, grid_size, lower_bound, upper_bound,
      std::numeric_limits<double>::epsilon() * 1.e8);

  // Test the four-index constraint with random numbers
  test_four_index_constraint_random<DataVector, 3, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_four_index_constraint_random<double, 3, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_four_index_constraint_random<double, 3, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());
}

void f_constraint() {
  // Test the F constraint against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> dimensionless_spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, dimensionless_spin, center);

  const size_t grid_size = 5;
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};

  // Note: looser numerical tolerance because this check
  // uses numerical derivatives
  test_f_constraint_analytic(solution, grid_size, lower_bound, upper_bound,
                             std::numeric_limits<double>::epsilon() * 1.e8);

  // Test the F constraint with random numbers
  test_f_constraint_random<DataVector, 1, Frame::Grid>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_f_constraint_random<DataVector, 1, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_f_constraint_random<double, 1, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_f_constraint_random<double, 1, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());

  test_f_constraint_random<DataVector, 2, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_f_constraint_random<double, 2, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_f_constraint_random<double, 2, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());

  test_f_constraint_random<DataVector, 3, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_f_constraint_random<double, 3, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_f_constraint_random<double, 3, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());
}

void constraint_energy() {
  // Test the constraint energy against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 4;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};

  test_constraint_energy_analytic(
      solution, grid_size, lower_bound, upper_bound,
      std::numeric_limits<double>::epsilon() * 1.e7);

  // Test the constraint energy with random numbers
  test_constraint_energy_random<DataVector, 1, Frame::Grid>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_constraint_energy_random<DataVector, 1, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_constraint_energy_random<double, 1, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_constraint_energy_random<double, 1, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());

  test_constraint_energy_random<DataVector, 2, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_constraint_energy_random<double, 2, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_constraint_energy_random<double, 2, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());

  test_constraint_energy_random<DataVector, 3, Frame::Inertial>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_constraint_energy_random<double, 3, Frame::Grid>(
      std::numeric_limits<double>::signaling_NaN());
  test_constraint_energy_random<double, 3, Frame::Inertial>(
      std::numeric_limits<double>::signaling_NaN());
}

void test_compute_tags() {
  // Test the F constraint against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> dimensionless_spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, dimensionless_spin, center);

  const size_t grid_size = 3;
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};

  test_constraint_compute_items(solution, grid_size, lower_bound, upper_bound);
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.Constraints",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};
  three_index_constraint();
  gauge_constraint();
  two_index_constraint();
  four_index_constraint();
  f_constraint();
  constraint_energy();
  test_compute_tags();
}
