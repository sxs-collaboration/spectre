// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/AreaElement.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Elliptic/Systems/Xcts/Events/ObserveAdmIntegrals.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/SpecialRelativity/LorentzBoostMatrix.hpp"
#include "PointwiseFunctions/Xcts/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"

namespace {

/**
 * This functions tests the ADM integrals using a boosted Schwarzschild
 * solution in isotropic coordinates. To do this, we consider two frames:
 * - Inertial frame (unbarred): the one in which we have a black hole moving
 * with a `boost_speed`; and
 * - BH frame (barred): the one in which the black hole is stationary.
 *
 * Note that the Schwarzschild solution only applies in the barred frame.
 * Here's an outline of the steps taken in this function:
 * 1. Define inertial coordinates;
 * 2. Transform inertial coordinates into barred coordinates;
 * 3. Using barred coordinates, measure the Schwarzschild spacetime metric;
 * 4. Transform the barred spacetime metric into the inertial frame; and
 * 5. Decompose metric into inertial variables and compute integrals.
 */
void test_local_adm_integrals(const double& distance,
                              const std::vector<double>& prev_distances) {
  // Define black hole parameters.
  const double mass = 1;
  const double boost_speed = 0.5;
  const double lorentz_factor = 1. / sqrt(1. - square(boost_speed));
  const std::array<double, 3> boost_velocity{{0., 0., boost_speed}};

  // Get Schwarzschild solution in isotropic coordinates.
  const Xcts::Solutions::Schwarzschild solution(
      mass, Xcts::Solutions::SchwarzschildCoordinates::Isotropic);

  // Get Lorentz boost matrix.
  // Note that `-boost_velocity` gives us a matrix in which the upper index is
  // barred. That is, it converts vectors into the barred frame and one-forms
  // into the inertial frame.
  const auto boost_matrix = sr::lorentz_boost_matrix(-boost_velocity);

  // Set up domain.
  const size_t h_refinement = 1;
  const size_t p_refinement = 6;
  domain::creators::Sphere shell{
      /* inner_radius */ 2 * mass,
      /* outer_radius */ distance,
      /* interior */ domain::creators::Sphere::Excision{},
      /* initial_refinement */ h_refinement,
      /* initial_number_of_grid_points */ p_refinement + 1,
      /* use_equiangular_map */ true,
      /* equatorial_compression */ {},
      /* radial_partitioning */ prev_distances,
      /* radial_distribution */ domain::CoordinateMaps::Distribution::Inverse};
  const auto shell_domain = shell.create_domain();
  const auto& blocks = shell_domain.blocks();
  const auto& initial_ref_levels = shell.initial_refinement_levels();
  const auto element_ids = initial_element_ids(initial_ref_levels);
  const Mesh<3> mesh{p_refinement + 1, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Mesh<2> face_mesh{p_refinement + 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  // Initialize "reduced" integrals.
  Scalar<double> total_adm_mass;
  total_adm_mass.get() = 0.;
  tnsr::I<double, 3> total_adm_linear_momentum;
  tnsr::I<double, 3> total_center_of_mass;
  for (int I = 0; I < 3; I++) {
    total_adm_linear_momentum.get(I) = 0.;
    total_center_of_mass.get(I) = 0.;
  }

  // Compute integral by summing over each element.
  for (const auto& element_id : element_ids) {
    // Get element information.
    const auto& current_block = blocks.at(element_id.block_id());
    const auto current_element = domain::Initialization::create_initial_element(
        element_id, current_block, initial_ref_levels);
    const ElementMap<3, Frame::Inertial> logical_to_inertial_map(
        element_id, current_block.stationary_map().get_clone());

    // Get inertial coordinates.
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = logical_to_inertial_map(logical_coords);
    const auto inv_jacobian =
        logical_to_inertial_map.inv_jacobian(logical_coords);

    // Transform coordinates into the barred frame.
    auto barred_coords =
        make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(
            inertial_coords, 0.0);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        barred_coords.get(i) +=
            boost_matrix.get(i + 1, j + 1) * inertial_coords.get(j);
      }
    }

    // Get barred spacetime variables.
    const auto barred_spacetime_vars = solution.variables(
        barred_coords,
        tmpl::list<gr::Tags::Lapse<DataVector>,
                   gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
                   gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>{});
    const auto& barred_lapse =
        get<gr::Tags::Lapse<DataVector>>(barred_spacetime_vars);
    const auto& barred_shift =
        get<gr::Tags::Shift<DataVector, 3, Frame::Inertial>>(
            barred_spacetime_vars);
    const auto& barred_spatial_metric =
        get<gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>(
            barred_spacetime_vars);

    // Construct barred spacetime metric.
    tnsr::aa<DataVector, 3, Frame::Inertial> barred_spacetime_metric;
    gr::spacetime_metric(make_not_null(&barred_spacetime_metric),
                         barred_lapse, barred_shift, barred_spatial_metric);

    // Transform spacetime metric into the inertial frame.
    auto spacetime_metric =
        make_with_value<tnsr::aa<DataVector, 3, Frame::Inertial>>(
            inertial_coords, 0.0);
    for (int a = 0; a < 4; a++) {
      for (int b = 0; b <= a; b++) {
        for (int c = 0; c < 4; c++) {
          for (int d = 0; d < 4; d++) {
            spacetime_metric.get(a, b) += boost_matrix.get(c, a) *
                                          boost_matrix.get(d, b) *
                                          barred_spacetime_metric.get(c, d);
          }
        }
      }
    }

    // Do the 3+1 decomposition.
    const auto spatial_metric = gr::spatial_metric(spacetime_metric);
    const auto& inv_spatial_metric =
        determinant_and_inverse(spatial_metric).second;
    const auto shift = gr::shift(spacetime_metric, inv_spatial_metric);
    const auto lapse = gr::lapse(shift, spacetime_metric);

    // Do the conformal decomposition.
    // Note that we choose the same conformal factor as the one used for the
    // Schwarzschild solution in isotropic coordinates. Other conformal factors
    // could also work.
    const auto solution_conformal_factor = solution.variables(
        barred_coords, tmpl::list<Xcts::Tags::ConformalFactor<DataVector>>{});
    const auto& conformal_factor =
        get<Xcts::Tags::ConformalFactor<DataVector>>(solution_conformal_factor);
    const auto conformal_metric = tenex::evaluate<ti::i, ti::j>(
        spatial_metric(ti::i, ti::j) / pow<4>(conformal_factor()));
    const auto inv_conformal_metric = tenex::evaluate<ti::I, ti::J>(
        inv_spatial_metric(ti::I, ti::J) * pow<4>(conformal_factor()));

    // Compute spatial derivatives.
    const auto deriv_shift = partial_derivative(shift, mesh, inv_jacobian);
    const auto deriv_spatial_metric =
        partial_derivative(spatial_metric, mesh, inv_jacobian);
    const auto deriv_conformal_factor =
        partial_derivative(conformal_factor, mesh, inv_jacobian);
    const auto deriv_conformal_metric =
        partial_derivative(conformal_metric, mesh, inv_jacobian);

    // Compute conformal Christoffel symbols.
    const auto conformal_christoffel_second_kind = gr::christoffel_second_kind(
        deriv_conformal_metric, inv_conformal_metric);
    const auto conformal_christoffel_contracted = tenex::evaluate<ti::i>(
        conformal_christoffel_second_kind(ti::J, ti::i, ti::j));

    // Define variables that appear in the formulas of dt_spatial_metric.
    const auto& x = get<0>(inertial_coords);
    const auto& y = get<1>(inertial_coords);
    const auto& z = get<2>(inertial_coords);
    const auto barred_r =
        sqrt(square(x) + square(y) + square(lorentz_factor * z));

    // Compute spatial metric time derivative.
    // Note that these formulas were derived in a Mathematica notebook
    // specifically for this problem. Here, we are evaluating them at t = 0.
    auto dt_spatial_metric =
        make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(
            inertial_coords, 0.0);
    dt_spatial_metric.get(0, 0) =
        (2 * mass * boost_speed * z * cube(1 + mass / barred_r)) /
        ((1 - square(boost_speed)) * cube(barred_r));
    dt_spatial_metric.get(1, 1) =
        (2 * mass * boost_speed * z * cube(1 + mass / barred_r)) /
        ((1 - square(boost_speed)) * cube(barred_r));
    dt_spatial_metric.get(2, 2) =
        (2 * mass * boost_speed * z * cube(1 + mass / barred_r)) /
            (square(1 - square(boost_speed)) * cube(barred_r)) +
        (2 * mass * cube(boost_speed) * z * cube(1 + mass / barred_r)) /
            (square(1 - square(boost_speed)) * cube(barred_r));

    // Compute extrinsic curvature and its trace.
    const auto extrinsic_curvature =
        gr::extrinsic_curvature(lapse, shift, deriv_shift, spatial_metric,
                                dt_spatial_metric, deriv_spatial_metric);
    const auto trace_extrinsic_curvature = tenex::evaluate(
        inv_spatial_metric(ti::I, ti::J) * extrinsic_curvature(ti::i, ti::j));

    // Compute face normal (related to the conformal metric).
    auto direction = Direction<3>::upper_zeta();
    auto conformal_face_normal =
        unnormalized_face_normal(face_mesh, logical_to_inertial_map, direction);
    const auto& face_inv_conformal_metric =
        dg::project_tensor_to_boundary(inv_conformal_metric, mesh, direction);
    const auto face_normal_magnitude =
        magnitude(conformal_face_normal, face_inv_conformal_metric);
    for (size_t d = 0; d < 3; ++d) {
      conformal_face_normal.get(d) /= get(face_normal_magnitude);
    }
    const DirectionMap<3, tnsr::i<DataVector, 3>> conformal_face_normals(
        {std::make_pair(direction, conformal_face_normal)});

    // Compute local integrals.
    Scalar<double> local_adm_mass;
    tnsr::I<double, 3> local_adm_linear_momentum;
    tnsr::I<double, 3> local_center_of_mass;
    Events::local_adm_integrals(
        make_not_null(&local_adm_mass),
        make_not_null(&local_adm_linear_momentum),
        make_not_null(&local_center_of_mass), conformal_factor,
        deriv_conformal_factor, conformal_metric, inv_conformal_metric,
        conformal_christoffel_second_kind, conformal_christoffel_contracted,
        spatial_metric, inv_spatial_metric, extrinsic_curvature,
        trace_extrinsic_curvature, inertial_coords, inv_jacobian, mesh,
        current_element, conformal_face_normals);
    total_adm_mass.get() += get(local_adm_mass);
    for (int I = 0; I < 3; I++) {
      total_adm_linear_momentum.get(I) += local_adm_linear_momentum.get(I);
      total_center_of_mass.get(I) += local_center_of_mass.get(I);
    }
  }

  // Check result
  auto custom_approx = Approx::custom().epsilon(10. / distance).scale(1.0);
  CHECK(get(total_adm_mass) == custom_approx(lorentz_factor * mass));
  CHECK(get<0>(total_adm_linear_momentum) == custom_approx(0.));
  CHECK(get<1>(total_adm_linear_momentum) == custom_approx(0.));
  CHECK(get<2>(total_adm_linear_momentum) ==
        custom_approx(lorentz_factor * mass * boost_speed));
  CHECK(get<0>(total_center_of_mass) == custom_approx(0.));
  CHECK(get<1>(total_center_of_mass) == custom_approx(0.));
  CHECK(get<2>(total_center_of_mass) == custom_approx(0.));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.ObserveAdmIntegrals",
                  "[Unit][PointwiseFunctions]") {
  // Test convergence with distance
  std::vector<double> prev_distances = {};
  for (const double distance : std::array<double, 3>{{1.e3, 1.e4, 1.e5}}) {
    test_local_adm_integrals(distance, prev_distances);
    prev_distances.push_back(distance);
  }
}
