// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestingFramework.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/Xcts/SpacetimeQuantities.hpp"

namespace TestHelpers::Xcts::Solutions {

namespace detail {

// Verify the Hamiltonian and momentum constraints are satisfied. This is only a
// consistency check of the solution, not a test of the system.
template <typename Solution>
void verify_adm_constraints(const Solution& solution,
                            const std::array<double, 3>& center,
                            const double inner_radius,
                            const double outer_radius, const double tolerance) {
  INFO("ADM constraints");
  Approx custom_approx = Approx::custom().epsilon(tolerance).scale(1.0);

  // Set up a grid for evaluating the solution and taking numerical derivatives
  const Mesh<3> mesh{12, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t num_points = mesh.number_of_grid_points();
  using AffineMap = domain::CoordinateMaps::Affine;
  using AffineMap3D =
      domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
  const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                              AffineMap3D>
      coord_map{
          {{-1., 1., center[0] + inner_radius, center[0] + outer_radius},
           {-1., 1., center[1] + inner_radius, center[1] + outer_radius},
           {-1., 1., center[2] + inner_radius, center[2] + outer_radius}}};
  const auto logical_coords = logical_coordinates(mesh);
  const auto x = coord_map(logical_coords);
  const auto inv_jacobian = coord_map.inv_jacobian(logical_coords);

  // Retrieve analytic variables
  using analytic_tags = tmpl::list<
      ::Xcts::Tags::ConformalFactor<DataVector>,
      ::Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
      gr::Tags::Shift<3, Frame::Inertial, DataVector>,
      gr::Tags::Lapse<DataVector>,
      ::Xcts::Tags::LapseTimesConformalFactor<DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Xcts::Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>,
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>,
      gr::Tags::Conformal<
          gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>, 0>>;
  const auto analytic_vars = solution.variables(x, analytic_tags{});
  const auto& conformal_factor =
      get<::Xcts::Tags::ConformalFactor<DataVector>>(analytic_vars);
  const auto& conformal_metric =
      get<::Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
          analytic_vars);
  const auto& inv_conformal_metric =
      get<::Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
          analytic_vars);
  const auto& lapse_times_conformal_factor =
      get<::Xcts::Tags::LapseTimesConformalFactor<DataVector>>(analytic_vars);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(analytic_vars);
  const auto& shift_excess =
      get<::Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
          analytic_vars);
  const auto& shift_background =
      get<::Xcts::Tags::ShiftBackground<DataVector, 3, Frame::Inertial>>(
          analytic_vars);
  const auto& shift =
      get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(analytic_vars);
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>(
          analytic_vars);
  const auto& trace_extrinsic_curvature =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(analytic_vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
          analytic_vars);
  const auto& inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>(
          analytic_vars);
  const auto& deriv_spatial_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>(analytic_vars);
  const auto& longitudinal_shift_excess = get<
      ::Xcts::Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>>(
      analytic_vars);
  const auto& longitudinal_shift_background_minus_dt_conformal_metric =
      get<::Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>>(analytic_vars);
  const auto& energy_density =
      get<gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>>(
          analytic_vars);
  const auto& momentum_density = get<gr::Tags::Conformal<
      gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>, 0>>(
      analytic_vars);

  // Compute Christoffels and Ricci
  const auto spatial_christoffel =
      gr::christoffel_second_kind(deriv_spatial_metric, inv_spatial_metric);
  const auto deriv_spatial_christoffel =
      partial_derivative(spatial_christoffel, mesh, inv_jacobian);
  const auto spatial_ricci =
      gr::ricci_tensor(spatial_christoffel, deriv_spatial_christoffel);
  const auto ricci_scalar = trace(spatial_ricci, inv_spatial_metric);

  // Check some identities
  CHECK_ITERABLE_APPROX(trace(extrinsic_curvature, inv_spatial_metric),
                        trace_extrinsic_curvature);
  CHECK_ITERABLE_APPROX(get(lapse) * get(conformal_factor),
                        get(lapse_times_conformal_factor));
  CHECK_ITERABLE_APPROX(TensorExpressions::evaluate<ti_I>(
                            shift_background(ti_I) + shift_excess(ti_I)),
                        shift);

  // Check extrinsic curvature decomposition
  //   K_ij = \psi^-2 \bar{A}_ij + 1/3 \gamma_ij K
  // with
  //   \bar{A}^ij = \psi^6 / (2 \alpha) * (\bar{L}\beta^ij - \bar{u}^ij)
  tnsr::ii<DataVector, 3> composed_extcurv{};
  TensorExpressions::evaluate<ti_i, ti_j>(
      make_not_null(&composed_extcurv),
      pow<4>(conformal_factor()) / (2. * lapse()) *
              (longitudinal_shift_excess(ti_K, ti_L) +
               longitudinal_shift_background_minus_dt_conformal_metric(ti_K,
                                                                       ti_L)) *
              conformal_metric(ti_i, ti_k) * conformal_metric(ti_l, ti_j) +
          spatial_metric(ti_i, ti_j) * trace_extrinsic_curvature() / 3.);
  CHECK_ITERABLE_APPROX(composed_extcurv, extrinsic_curvature);

  // Check Hamiltonian constraint R + K^2 + K_ij K^ij = 16 \pi \rho
  const Scalar<DataVector> hamiltonian_constraint = TensorExpressions::evaluate(
      ricci_scalar() + square(trace_extrinsic_curvature()) -
      extrinsic_curvature(ti_i, ti_j) * extrinsic_curvature(ti_k, ti_l) *
          inv_spatial_metric(ti_I, ti_K) * inv_spatial_metric(ti_J, ti_L) -
      16. * M_PI * energy_density());
  CHECK_ITERABLE_CUSTOM_APPROX(get(hamiltonian_constraint),
                               DataVector(num_points, 0.), custom_approx);

  // Check momentum constraint \nabla_j (K^ij - K \gamma^ij) = 8 \pi S^i
  tnsr::II<DataVector, 3> extcurv_min_trace{};
  TensorExpressions::evaluate<ti_I, ti_J>(
      make_not_null(&extcurv_min_trace),
      extrinsic_curvature(ti_k, ti_l) * inv_spatial_metric(ti_I, ti_K) *
              inv_spatial_metric(ti_J, ti_L) -
          trace_extrinsic_curvature() * inv_spatial_metric(ti_I, ti_J));
  const tnsr::iJJ<DataVector, 3> deriv_extcurv_min_trace =
      partial_derivative(extcurv_min_trace, mesh, inv_jacobian);
  const tnsr::I<DataVector, 3> momentum_constraint =
      TensorExpressions::evaluate<ti_I>(
          deriv_extcurv_min_trace(ti_j, ti_I, ti_J) +
          spatial_christoffel(ti_I, ti_k, ti_j) *
              extcurv_min_trace(ti_K, ti_J) +
          spatial_christoffel(ti_J, ti_k, ti_j) *
              extcurv_min_trace(ti_I, ti_K) -
          8. * M_PI * momentum_density(ti_I));
  CHECK_ITERABLE_CUSTOM_APPROX(momentum_constraint,
                               (tnsr::I<DataVector, 3>(num_points, 0.)),
                               custom_approx);

  // Test things yet another way: compute quantities from XCTS variables alone
  // (as they would come out of the elliptic solver) and compare to analytic
  // solution
  ::Xcts::SpacetimeQuantities spacetime_quantities{num_points};
  const ::Xcts::SpacetimeQuantitiesComputer computer{
      conformal_factor,
      lapse_times_conformal_factor,
      shift_excess,
      conformal_metric,
      inv_conformal_metric,
      shift_background,
      mesh,
      inv_jacobian};
  const auto get_var = [&spacetime_quantities, &computer](const auto tag) {
    return spacetime_quantities.get_var(computer, tag);
  };
  // Quantities that involve numeric derivatives are compared with the given
  // `tolerance`
  CHECK_ITERABLE_APPROX(get_var(gr::Tags::Lapse<DataVector>{}), lapse);
  CHECK_ITERABLE_APPROX(
      get_var(gr::Tags::Shift<3, Frame::Inertial, DataVector>{}), shift);
  CHECK_ITERABLE_APPROX(
      get_var(gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>{}),
      spatial_metric);
  CHECK_ITERABLE_APPROX(
      get_var(gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>{}),
      inv_spatial_metric);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_var(
          ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>{}),
      deriv_spatial_metric, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_var(gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>{}),
      extrinsic_curvature, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_var(gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial,
                                                     DataVector>{}),
      spatial_christoffel, custom_approx);
  // These second derivatives (and dependent quantities) exceed the standard
  // tolerance
  Approx custom_approx2 = Approx::custom().epsilon(tolerance * 100).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_var(::Tags::deriv<gr::Tags::SpatialChristoffelSecondKind<
                                3, Frame::Inertial, DataVector>,
                            tmpl::size_t<3>, Frame::Inertial>{}),
      deriv_spatial_christoffel, custom_approx2);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_var(gr::Tags::SpatialRicci<3, Frame::Inertial, DataVector>{}),
      spatial_ricci, custom_approx2);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_var(gr::Tags::HamiltonianConstraint<DataVector>{}),
      Scalar<DataVector>(num_points, 0.), custom_approx2);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get_var(gr::Tags::MomentumConstraint<3, Frame::Inertial, DataVector>{}),
      (tnsr::i<DataVector, 3>(num_points, 0.)), custom_approx2);
}

template <::Xcts::Equations EnabledEquations,
          ::Xcts::Geometry ConformalGeometry, int ConformalMatterScale,
          typename Solution>
void verify_solution_impl(const Solution& solution,
                          const std::array<double, 3>& center,
                          const double inner_radius, const double outer_radius,
                          const double tolerance) {
  CAPTURE(EnabledEquations);
  CAPTURE(ConformalGeometry);
  using system = ::Xcts::FirstOrderSystem<EnabledEquations, ConformalGeometry,
                                          ConformalMatterScale>;
  const Mesh<3> mesh{12, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  using AffineMap = domain::CoordinateMaps::Affine;
  using AffineMap3D =
      domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
  const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                              AffineMap3D>
      coord_map{
          {{-1., 1., center[0] + inner_radius, center[0] + outer_radius},
           {-1., 1., center[1] + inner_radius, center[1] + outer_radius},
           {-1., 1., center[2] + inner_radius, center[2] + outer_radius}}};
  FirstOrderEllipticSolutionsTestHelpers::verify_solution<system>(
      solution, mesh, coord_map, tolerance);
}
}  // namespace detail

/*!
 * \brief Verify the `solution` solves the XCTS equations numerically.
 *
 * \tparam ConformalGeometry Specify `Xcts::Geometry::FlatCartesian` to test
 * both the flat _and_ the curved system, or `Xcts::Geometry::Curved` to only
 * test the curved system (for solutions on a curved conformal background).
 * \tparam Solution The analytic solution to test (inferred)
 * \param solution The analytic solution to test
 * \param center offset for the \p inner_radius and the \p outer_radius
 * \param inner_radius Lower-left corner of a cube on which to test
 * \param outer_radius Upper-right corner of a cube on which to test
 * \param tolerance Requested tolerance
 */
template <::Xcts::Geometry ConformalGeometry, int ConformalMatterScale,
          typename Solution>
void verify_solution(const Solution& solution,
                     const std::array<double, 3>& center,
                     const double inner_radius, const double outer_radius,
                     const double tolerance) {
  CAPTURE(tolerance);
  detail::verify_adm_constraints(solution, center, inner_radius, outer_radius,
                                 tolerance);
  if constexpr (ConformalGeometry == ::Xcts::Geometry::FlatCartesian) {
    INVOKE_TEST_FUNCTION(
        detail::verify_solution_impl,
        (solution, center, inner_radius, outer_radius, tolerance),
        (::Xcts::Equations::Hamiltonian, ::Xcts::Equations::HamiltonianAndLapse,
         ::Xcts::Equations::HamiltonianLapseAndShift),
        (::Xcts::Geometry::FlatCartesian, ::Xcts::Geometry::Curved),
        (ConformalMatterScale));
  } else {
    INVOKE_TEST_FUNCTION(
        detail::verify_solution_impl,
        (solution, center, inner_radius, outer_radius, tolerance),
        (::Xcts::Equations::Hamiltonian, ::Xcts::Equations::HamiltonianAndLapse,
         ::Xcts::Equations::HamiltonianLapseAndShift),
        (::Xcts::Geometry::Curved), (ConformalMatterScale));
  }
}

}  // namespace TestHelpers::Xcts::Solutions
