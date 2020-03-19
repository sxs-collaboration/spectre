// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
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
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare Tags::deriv

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
void test_three_index_constraint(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iaa<DataType, SpatialDim, Frame> (*)(
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &GeneralizedHarmonic::three_index_constraint<SpatialDim, Frame,
                                                       DataType>),
      "numpy", "subtract", {{{-1.0, 1.0}}}, used_for_size);
}

// Test the return-by-value gauge constraint function using random values
template <size_t SpatialDim, typename Frame, typename DataType>
void test_gauge_constraint_random(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::AA<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &GeneralizedHarmonic::gauge_constraint<SpatialDim, Frame, DataType>),
      "TestFunctions", "gauge_constraint", {{{-1.0, 1.0}}}, used_for_size);
}

// Test the return-by-reference gauge constraint by comparing to Kerr-Schild
template <typename Solution>
void test_gauge_constraint_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Check vs. time-independent analytic solution
  // Set up grid
  Mesh<3> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
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
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<3>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<3>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<3>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<3>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients for computing the gauge constraint
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto gauge_function = GeneralizedHarmonic::gauge_source(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, spatial_metric,
      trace(gr::extrinsic_curvature(lapse, shift, d_shift, spatial_metric,
                                    dt_spatial_metric, d_spatial_metric),
            inverse_spatial_metric),
      trace_last_indices(gr::christoffel_first_kind(d_spatial_metric),
                         inverse_spatial_metric));
  const auto phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                            spatial_metric, d_spatial_metric);
  const auto pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<3, Frame::Inertial>(lapse);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);

  // Get the constraint, and check that it vanishes
  auto constraint =
      make_with_value<tnsr::a<DataVector, 3, Frame::Inertial>>(x, 0.0);
  const gsl::not_null<tnsr::a<DataVector, 3, Frame::Inertial>*>
      constraint_pointer = &(constraint);
  GeneralizedHarmonic::gauge_constraint(
      constraint_pointer, gauge_function, normal_one_form, normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi);
  CHECK_ITERABLE_APPROX(constraint,
                        make_with_value<decltype(constraint)>(x, 0.0));
}

// Test the return-by-value two-index constraint function using random values
template <size_t SpatialDim, typename Frame, typename DataType>
void test_two_index_constraint_random(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ia<DataType, SpatialDim, Frame> (*)(
          const tnsr::ia<DataType, SpatialDim, Frame>&,
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
          &GeneralizedHarmonic::two_index_constraint<SpatialDim, Frame,
                                                     DataType>),
      "TestFunctions", "two_index_constraint", {{{-1.0, 1.0}}},
      used_for_size);
}

// Test the return-by-reference two-index constraint
// by comparing to a time-independent analytic solution (e.g. Kerr-Schild)
template <typename Solution>
void test_two_index_constraint_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound,
    const double error_tolerance) noexcept {
  // Shorter names for tags.
  using SpacetimeMetric = gr::Tags::SpacetimeMetric<3, Frame::Inertial>;
  using Pi = ::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>;
  using Phi = ::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>;
  using GaugeH = ::GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>;
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
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
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
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<3>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<3>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<3>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<3>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Compute quantities from variables for the two-index constraint
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<3, Frame::Inertial>(lapse);
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
  phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift, spatial_metric,
                                 d_spatial_metric);
  pi = GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
                               dt_spatial_metric, phi);
  gauge_function = GeneralizedHarmonic::gauge_source(
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

  // Compute the three-index constraint
  const auto three_index_constraint =
      GeneralizedHarmonic::three_index_constraint(d_spacetime_metric, phi);

  // Get the constraint, and check that it vanishes to error_tolerance
  auto two_index_constraint =
      make_with_value<tnsr::ia<DataVector, 3, Frame::Inertial>>(
          x, std::numeric_limits<double>::signaling_NaN());
  GeneralizedHarmonic::two_index_constraint(
      make_not_null(&two_index_constraint), d_gauge_function, normal_one_form,
      normal_vector, inverse_spatial_metric, inverse_spacetime_metric, pi, phi,
      d_pi, d_phi, gamma2, three_index_constraint);

  Approx numerical_approx =
      Approx::custom().epsilon(error_tolerance).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      two_index_constraint,
      make_with_value<decltype(two_index_constraint)>(x, 0.0),
      numerical_approx);
}

// Test the return-by-value four-index constraint function using random values
template <size_t SpatialDim, typename Frame, typename DataType>
void test_four_index_constraint_random(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iaa<DataType, SpatialDim, Frame> (*)(
          const tnsr::ijaa<DataType, SpatialDim, Frame>&)>(
          &GeneralizedHarmonic::four_index_constraint<SpatialDim, Frame,
                                                      DataType>),
      "TestFunctions", "four_index_constraint", {{{-1.0, 1.0}}},
      used_for_size);
}

// Test the return-by-reference four-index constraint
// by comparing to a time-independent analytic solution (e.g. Kerr-Schild)
template <typename Solution>
void test_four_index_constraint_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound,
    const double error_tolerance) noexcept {
  // Shorter names for tags.
  using Phi = ::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>;
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
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
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
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<3>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<3>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Compute derivative d_phi numerically
  Variables<VariablesTags> gh_vars(data_size);
  get<Phi>(gh_vars) = GeneralizedHarmonic::phi(
      lapse, d_lapse, shift, d_shift, spatial_metric, d_spatial_metric);

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
  GeneralizedHarmonic::four_index_constraint(
      make_not_null(&four_index_constraint), d_phi);

  Approx numerical_approx =
      Approx::custom().epsilon(error_tolerance).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      four_index_constraint,
      make_with_value<decltype(four_index_constraint)>(x, 0.0),
      numerical_approx);
}

// Test the return-by-value F constraint function using random values
template <size_t SpatialDim, typename Frame, typename DataType>
void test_f_constraint_random(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::ia<DataType, SpatialDim, Frame>&,
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
          &GeneralizedHarmonic::f_constraint<SpatialDim, Frame, DataType>),
      "TestFunctions", "f_constraint", {{{-1.0, 1.0}}}, used_for_size);
}

// Test the return-by-reference F constraint
// by comparing to a time-independent analytic solution (e.g. Kerr-Schild)
template <typename Solution>
void test_f_constraint_analytic(const Solution& solution,
                                const size_t grid_size_each_dimension,
                                const std::array<double, 3>& lower_bound,
                                const std::array<double, 3>& upper_bound,
                                const double error_tolerance) noexcept {
  // Shorter names for tags.
  using SpacetimeMetric = gr::Tags::SpacetimeMetric<3, Frame::Inertial>;
  using Pi = ::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>;
  using Phi = ::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>;
  using GaugeH = ::GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>;
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
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
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
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<3>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<3>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<3>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<3>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Compute quantities from variables for the two-index constraint
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<3, Frame::Inertial>(lapse);
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
  phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift, spatial_metric,
                                 d_spatial_metric);
  pi = GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
                               dt_spatial_metric, phi);
  gauge_function = GeneralizedHarmonic::gauge_source(
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

  // Compute the three-index constraint
  const auto three_index_constraint =
      GeneralizedHarmonic::three_index_constraint(d_spacetime_metric, phi);

  // Get the constraint, and check that it vanishes to error_tolerance
  auto f_constraint = make_with_value<tnsr::a<DataVector, 3, Frame::Inertial>>(
      x, std::numeric_limits<double>::signaling_NaN());
  GeneralizedHarmonic::f_constraint(
      make_not_null(&f_constraint), gauge_function, d_gauge_function,
      normal_one_form, normal_vector, inverse_spatial_metric,
      inverse_spacetime_metric, pi, phi, d_pi, d_phi, gamma2,
      three_index_constraint);

  Approx numerical_approx =
      Approx::custom().epsilon(error_tolerance).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(f_constraint,
                               make_with_value<decltype(f_constraint)>(x, 0.0),
                               numerical_approx);
}

// Test the return-by-value constraint energy function using random values
template <size_t SpatialDim, typename Frame, typename DataType>
void test_constraint_energy_random(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::ia<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&, const Scalar<DataType>&,
          double, double, double, double)>(
          &GeneralizedHarmonic::constraint_energy<SpatialDim, Frame, DataType>),
      "TestFunctions", "constraint_energy", {{{-1.0, 1.0}}},
      used_for_size);
}

// Test the return-by-reference constraint energy
// by comparing to a time-independent analytic solution (e.g. Kerr-Schild)
template <typename Solution>
void test_constraint_energy_analytic(const Solution& solution,
                                     const size_t grid_size_each_dimension,
                                     const std::array<double, 3>& lower_bound,
                                     const std::array<double, 3>& upper_bound,
                                     const double error_tolerance) noexcept {
  // Shorter names for tags.
  using SpacetimeMetric = gr::Tags::SpacetimeMetric<3, Frame::Inertial>;
  using Pi = ::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>;
  using Phi = ::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>;
  using GaugeH = ::GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>;
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
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
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
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& shift = get<gr::Tags::Shift<3>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<3>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<3>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<3>>>(vars);

  const auto determinant_and_inverse_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& determinant_spatial_metric =
      determinant_and_inverse_spatial_metric.first;
  const auto& inverse_spatial_metric =
      determinant_and_inverse_spatial_metric.second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<3, Frame::Inertial>(lapse);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);

  // Compute derivative d_phi numerically
  Variables<VariablesTags> gh_vars(data_size);
  auto& spacetime_metric = get<SpacetimeMetric>(gh_vars);
  auto& pi = get<Pi>(gh_vars);
  auto& phi = get<Phi>(gh_vars);
  auto& gauge_function = get<GaugeH>(gh_vars);
  spacetime_metric = gr::spacetime_metric(lapse, shift, spatial_metric);
  phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift, spatial_metric,
                                 d_spatial_metric);
  pi = GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift, spatial_metric,
                               dt_spatial_metric, phi);
  gauge_function = GeneralizedHarmonic::gauge_source(
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

  // Arbitrary choice for gamma2
  const auto gamma2 = make_with_value<Scalar<DataVector>>(x, 4.0);

  // Get the constraints
  const auto gauge_constraint = GeneralizedHarmonic::gauge_constraint(
      gauge_function, normal_one_form, normal_vector, inverse_spatial_metric,
      inverse_spacetime_metric, pi, phi);
  const auto three_index_constraint =
      GeneralizedHarmonic::three_index_constraint(d_spacetime_metric, phi);
  const auto four_index_constraint =
      GeneralizedHarmonic::four_index_constraint(d_phi);
  const auto two_index_constraint = GeneralizedHarmonic::two_index_constraint(
      d_gauge_function, normal_one_form, normal_vector, inverse_spatial_metric,
      inverse_spacetime_metric, pi, phi, d_pi, d_phi, gamma2,
      three_index_constraint);
  const auto f_constraint = GeneralizedHarmonic::f_constraint(
      gauge_function, d_gauge_function, normal_one_form, normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi, d_pi, d_phi,
      gamma2, three_index_constraint);

  auto constraint_energy = make_with_value<Scalar<DataVector>>(
      x, std::numeric_limits<double>::signaling_NaN());
  GeneralizedHarmonic::constraint_energy(
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
void test_constraint_compute_items(
    const Solution& solution, const size_t grid_size,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Check that compute items are named correctly
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::ConstraintGamma0Compute<3, Frame::Inertial>>(
      "ConstraintGamma0");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::ConstraintGamma1Compute<3, Frame::Inertial>>(
      "ConstraintGamma1");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::ConstraintGamma2Compute<3, Frame::Inertial>>(
      "ConstraintGamma2");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::GaugeHImplicitFrom3p1QuantitiesCompute<
          3, Frame::Inertial>>("GaugeH");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<3,
                                                             Frame::Inertial>>(
      "SpacetimeDerivGaugeH");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::GaugeConstraintCompute<3, Frame::Inertial>>(
      "GaugeConstraint");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::FConstraintCompute<3, Frame::Inertial>>(
      "FConstraint");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::TwoIndexConstraintCompute<3, Frame::Inertial>>(
      "TwoIndexConstraint");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::ThreeIndexConstraintCompute<3,
                                                             Frame::Inertial>>(
      "ThreeIndexConstraint");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::FourIndexConstraintCompute<3,
                                                            Frame::Inertial>>(
      "FourIndexConstraint");
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::Tags::ConstraintEnergyCompute<3, Frame::Inertial>>(
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
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
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
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& time_deriv_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& deriv_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<3>>(vars);
  const auto& deriv_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& time_deriv_shift = get<Tags::dt<gr::Tags::Shift<3>>>(vars);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<3>>(vars);
  const auto& time_deriv_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<3>>>(vars);
  const auto& deriv_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Compute quantities from variables for the two-index constraint
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);
  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<3, Frame::Inertial, DataVector>(lapse);
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& det_spatial_metric = det_and_inv.first;
  const auto& inverse_spatial_metric = det_and_inv.second;
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);

  const auto christoffel_first_kind =
      gr::christoffel_first_kind(deriv_spatial_metric);
  const auto trace_christoffel_first_kind =
      trace_last_indices(christoffel_first_kind, inverse_spatial_metric);

  // Compute derivatives d_phi, d_pi, and d_gauge_function numerically
  // First, prepare
  using VariablesTags =
      tmpl::list<GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
                 GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>>;

  Variables<VariablesTags> gh_vars(data_size);
  auto& pi = get<GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(gh_vars);
  auto& phi = get<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(gh_vars);
  auto& gauge_source =
      get<GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>>(gh_vars);
  phi = GeneralizedHarmonic::phi(lapse, deriv_lapse, shift, deriv_shift,
                                 spatial_metric, deriv_spatial_metric);
  pi = GeneralizedHarmonic::pi(lapse, time_deriv_lapse, shift, time_deriv_shift,
                               spatial_metric, time_deriv_spatial_metric, phi);
  const auto extrinsic_curvature = GeneralizedHarmonic::extrinsic_curvature(
      spacetime_normal_vector, pi, phi);
  const auto trace_extrinsic_curvature =
      trace(extrinsic_curvature, inverse_spatial_metric);
  gauge_source = GeneralizedHarmonic::gauge_source(
      lapse, time_deriv_lapse, deriv_lapse, shift, time_deriv_shift,
      deriv_shift, spatial_metric, trace_extrinsic_curvature,
      trace_christoffel_first_kind);
  // Second, compute derivatives
  const auto gh_derivs =
      partial_derivatives<VariablesTags, VariablesTags, 3, Frame::Inertial>(
          gh_vars, mesh, coord_map.inv_jacobian(x_logical));
  const auto& deriv_pi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);
  const auto& deriv_phi =
      get<Tags::deriv<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);
  const auto& deriv_gauge_source =
      get<Tags::deriv<GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);

  // Compute other derivatives
  const auto derivatives_of_spacetime_metric =
      gr::derivatives_of_spacetime_metric(
          lapse, time_deriv_lapse, deriv_lapse, shift, time_deriv_shift,
          deriv_shift, spatial_metric, time_deriv_spatial_metric,
          deriv_spatial_metric);
  tnsr::iaa<DataVector, 3, Frame::Inertial> deriv_spacetime_metric{};
  gr::Tags::DerivSpacetimeMetricCompute<3, Frame::Inertial>::function(
      make_not_null(&deriv_spacetime_metric), derivatives_of_spacetime_metric);

  auto time_deriv_gauge_source =
      make_with_value<tnsr::a<DataVector, 3, Frame::Inertial>>(x, 0.);
  get<0>(time_deriv_gauge_source) = 0.05;
  get<1>(time_deriv_gauge_source) = 0.06;
  get<2>(time_deriv_gauge_source) = 0.07;
  get<3>(time_deriv_gauge_source) = -0.05;

  const auto derivatives_of_gauge_source =
      GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<
          3, Frame::Inertial>::function(time_deriv_gauge_source,
                                        deriv_gauge_source);

  // Insert into databox
  const auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Coordinates<3, Frame::Inertial>,
          gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
          gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<3, Frame::Inertial, DataVector>,
          ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>,
          ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>,
          ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>,
          ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>,
          ::Tags::dt<gr::Tags::Lapse<DataVector>>,
          ::Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataVector>>,
          ::Tags::deriv<GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>,
                        tmpl::size_t<3>, Frame::Inertial>,
          ::Tags::dt<GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>>,
          ::Tags::deriv<GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>,
                        tmpl::size_t<3>, Frame::Inertial>,
          ::Tags::deriv<GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>,
                        tmpl::size_t<3>, Frame::Inertial>>,
      db::AddComputeTags<
          GeneralizedHarmonic::Tags::ConstraintGamma0Compute<3,
                                                             Frame::Inertial>,
          GeneralizedHarmonic::Tags::ConstraintGamma1Compute<3,
                                                             Frame::Inertial>,
          GeneralizedHarmonic::Tags::ConstraintGamma2Compute<3,
                                                             Frame::Inertial>,
          gr::Tags::SpacetimeNormalOneFormCompute<3, Frame::Inertial,
                                                  DataVector>,
          gr::Tags::SpacetimeNormalVectorCompute<3,
                                                 Frame::Inertial, DataVector>,
          gr::Tags::DetAndInverseSpatialMetricCompute<3, Frame::Inertial,
                                                      DataVector>,
          gr::Tags::InverseSpacetimeMetricCompute<3, Frame::Inertial,
                                                  DataVector>,
          gr::Tags::SpatialChristoffelFirstKindCompute<3, Frame::Inertial,
                                                       DataVector>,
          gr::Tags::TraceSpatialChristoffelFirstKindCompute<3, Frame::Inertial,
                                                            DataVector>,
          GeneralizedHarmonic::Tags::PhiCompute<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::PiCompute<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::ExtrinsicCurvatureCompute<3,
                                                               Frame::Inertial>,
          GeneralizedHarmonic::Tags::TraceExtrinsicCurvatureCompute<
              3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::GaugeHImplicitFrom3p1QuantitiesCompute<
              3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<
              3, Frame::Inertial>,
          gr::Tags::DerivativesOfSpacetimeMetricCompute<3, Frame::Inertial>,
          gr::Tags::DerivSpacetimeMetricCompute<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::FourIndexConstraintCompute<
              3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::ThreeIndexConstraintCompute<
              3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::TwoIndexConstraintCompute<3,
                                                               Frame::Inertial>,
          GeneralizedHarmonic::Tags::GaugeConstraintCompute<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::FConstraintCompute<3, Frame::Inertial>,
          GeneralizedHarmonic::Tags::ConstraintEnergyCompute<3,
                                                             Frame::Inertial>>>(
      x, spatial_metric, lapse, shift, deriv_spatial_metric, deriv_lapse,
      deriv_shift, time_deriv_spatial_metric, time_deriv_lapse,
      time_deriv_shift, deriv_gauge_source, time_deriv_gauge_source, deriv_phi,
      deriv_pi);

  // Compute tested quantities locally
  const auto gamma0 = GeneralizedHarmonic::Tags::ConstraintGamma0Compute<
      3, Frame::Inertial>::function(x);
  const auto gamma1 = GeneralizedHarmonic::Tags::ConstraintGamma1Compute<
      3, Frame::Inertial>::function(x);
  const auto gamma2 = GeneralizedHarmonic::Tags::ConstraintGamma2Compute<
      3, Frame::Inertial>::function(x);

  const auto four_index_constraint =
      GeneralizedHarmonic::four_index_constraint(deriv_phi);
  const auto three_index_constraint =
      GeneralizedHarmonic::three_index_constraint(deriv_spacetime_metric, phi);
  const auto two_index_constraint = GeneralizedHarmonic::two_index_constraint(
      deriv_gauge_source, spacetime_normal_one_form, spacetime_normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi, deriv_pi,
      deriv_phi, gamma2, three_index_constraint);
  const auto gauge_constraint = GeneralizedHarmonic::gauge_constraint(
      gauge_source, spacetime_normal_one_form, spacetime_normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi);
  const auto f_constraint = GeneralizedHarmonic::f_constraint(
      gauge_source, deriv_gauge_source, spacetime_normal_one_form,
      spacetime_normal_vector, inverse_spatial_metric, inverse_spacetime_metric,
      pi, phi, deriv_pi, deriv_phi, gamma2, three_index_constraint);
  const auto constraint_energy = GeneralizedHarmonic::constraint_energy(
      gauge_constraint, f_constraint, two_index_constraint,
      three_index_constraint, four_index_constraint, inverse_spatial_metric,
      det_spatial_metric);

  // Check that their compute items in databox furnish identical values
  CHECK(db::get<GeneralizedHarmonic::Tags::ConstraintGamma0>(box) == gamma0);
  CHECK(db::get<GeneralizedHarmonic::Tags::ConstraintGamma1>(box) == gamma1);
  CHECK(db::get<GeneralizedHarmonic::Tags::ConstraintGamma2>(box) == gamma2);
  CHECK(db::get<GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>>(box) ==
        gauge_source);
  CHECK(
      db::get<
          GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<3, Frame::Inertial>>(
          box) == derivatives_of_gauge_source);
  CHECK(db::get<
            GeneralizedHarmonic::Tags::FourIndexConstraint<3, Frame::Inertial>>(
            box) == four_index_constraint);
  CHECK(
      db::get<
          GeneralizedHarmonic::Tags::ThreeIndexConstraint<3, Frame::Inertial>>(
          box) == three_index_constraint);
  CHECK(db::get<
            GeneralizedHarmonic::Tags::TwoIndexConstraint<3, Frame::Inertial>>(
            box) == two_index_constraint);
  CHECK(db::get<GeneralizedHarmonic::Tags::GaugeConstraint<3, Frame::Inertial>>(
            box) == gauge_constraint);
  CHECK(db::get<GeneralizedHarmonic::Tags::FConstraint<3, Frame::Inertial>>(
            box) == f_constraint);
  CHECK(
      db::get<GeneralizedHarmonic::Tags::ConstraintEnergy<3, Frame::Inertial>>(
          box) == constraint_energy);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.ThreeIndexConstraint",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  test_three_index_constraint<1, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<1, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<1, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<1, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_three_index_constraint<2, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<2, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<2, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<2, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_three_index_constraint<3, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<3, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<3, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<3, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.GaugeConstraint",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  // Test the gauge constraint against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};

  test_gauge_constraint_analytic(solution, grid_size, lower_bound, upper_bound);

  // Test the gauge constraint with random numbers
  test_gauge_constraint_random<1, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<1, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<1, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_gauge_constraint_random<1, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_gauge_constraint_random<2, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<2, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<2, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_gauge_constraint_random<2, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_gauge_constraint_random<3, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<3, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<3, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_gauge_constraint_random<3, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.TwoIndexConstraint",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  // Test the two-index constraint against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> dimensionless_spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, dimensionless_spin, center);

  const size_t grid_size = 8;
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};

  // Note: looser numerical tolerance because this check
  // uses numerical derivatives
  test_two_index_constraint_analytic(
      solution, grid_size, lower_bound, upper_bound,
      std::numeric_limits<double>::epsilon() * 1.e6);

  // Test the two-index constraint with random numbers
  test_two_index_constraint_random<1, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_two_index_constraint_random<1, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_two_index_constraint_random<1, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_two_index_constraint_random<1, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_two_index_constraint_random<2, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_two_index_constraint_random<2, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_two_index_constraint_random<2, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_two_index_constraint_random<2, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_two_index_constraint_random<3, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_two_index_constraint_random<3, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_two_index_constraint_random<3, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_two_index_constraint_random<3, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.FourIndexConstraint",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  // Test the four-index constraint against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> dimensionless_spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, dimensionless_spin, center);

  const size_t grid_size = 8;
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};

  // Note: looser numerical tolerance because this check
  // uses numerical derivatives
  test_four_index_constraint_analytic(
      solution, grid_size, lower_bound, upper_bound,
      std::numeric_limits<double>::epsilon() * 1.e6);

  // Test the four-index constraint with random numbers
  test_four_index_constraint_random<3, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_four_index_constraint_random<3, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_four_index_constraint_random<3, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_four_index_constraint_random<3, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.FConstraint",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};
  // Test the F constraint against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> dimensionless_spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, dimensionless_spin, center);

  const size_t grid_size = 6;
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};

  // Note: looser numerical tolerance because this check
  // uses numerical derivatives
  test_f_constraint_analytic(solution, grid_size, lower_bound, upper_bound,
                             std::numeric_limits<double>::epsilon() * 1.e6);

  // Test the F constraint with random numbers
  test_f_constraint_random<1, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_f_constraint_random<1, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_f_constraint_random<1, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_f_constraint_random<1, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_f_constraint_random<2, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_f_constraint_random<2, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_f_constraint_random<2, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_f_constraint_random<2, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_f_constraint_random<3, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_f_constraint_random<3, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_f_constraint_random<3, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_f_constraint_random<3, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.ConstraintEnergy",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};
  // Test the constraint energy against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 6;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};

  test_constraint_energy_analytic(
      solution, grid_size, lower_bound, upper_bound,
      std::numeric_limits<double>::epsilon() * 1.e6);

  // Test the constraint energy with random numbers
  test_constraint_energy_random<1, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_constraint_energy_random<1, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_constraint_energy_random<1, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_constraint_energy_random<1, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_constraint_energy_random<2, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_constraint_energy_random<2, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_constraint_energy_random<2, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_constraint_energy_random<2, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_constraint_energy_random<3, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_constraint_energy_random<3, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_constraint_energy_random<3, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_constraint_energy_random<3, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.ConstraintComputeTags",
    "[Unit][Evolution]") {
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
