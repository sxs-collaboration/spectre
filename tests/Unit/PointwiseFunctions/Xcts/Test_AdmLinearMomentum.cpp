// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/AreaElement.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "PointwiseFunctions/Xcts/AdmLinearMomentum.hpp"

namespace {

using Schwarzschild = Xcts::Solutions::Schwarzschild;
using KerrSchild = Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild>;

template <typename Solution>
void test_infinite_surface_integral(const double distance, const double mass,
                                    const double horizon_radius,
                                    const double boost_speed,
                                    const Solution& solution) {
  // Set up domain
  const size_t h_refinement = 1;
  const size_t p_refinement = 6;
  const domain::creators::Sphere shell{
      /* inner_radius */ horizon_radius,
      /* outer_radius */ distance,
      /* interior */ domain::creators::Sphere::Excision{},
      /* initial_refinement */ h_refinement,
      /* initial_number_of_grid_points */ p_refinement + 1,
      /* use_equiangular_map */ true,
      /* equatorial_compression */ {},
      /* radial_partitioning */ {},
      /* radial_distribution */ domain::CoordinateMaps::Distribution::Inverse};
  const auto shell_domain = shell.create_domain();
  const auto& blocks = shell_domain.blocks();
  const auto& initial_ref_levels = shell.initial_refinement_levels();
  const auto element_ids = initial_element_ids(initial_ref_levels);
  const Mesh<2> face_mesh{p_refinement + 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  // Initialize surface integral
  tnsr::I<double, 3> surface_integral({0., 0., 0.});

  // Compute integrals by summing over each element
  for (const auto& element_id : element_ids) {
    // Skip elements not at the outer boundary
    const size_t radial_dimension = 2;
    const auto radial_segment_id = element_id.segment_id(radial_dimension);
    if (radial_segment_id.index() !=
        two_to_the(radial_segment_id.refinement_level()) - 1) {
      continue;
    }

    // Get element information
    const auto& current_block = blocks.at(element_id.block_id());
    const auto current_element = domain::Initialization::create_initial_element(
        element_id, current_block, initial_ref_levels);
    const ElementMap<3, Frame::Inertial> logical_to_inertial_map(
        element_id, current_block.stationary_map().get_clone());

    // Loop over external boundaries
    for (auto boundary_direction : current_element.external_boundaries()) {
      // Skip interfaces not at the outer boundary
      if (boundary_direction != Direction<3>::upper_zeta()) {
        continue;
      }

      // Get coordinates
      const auto logical_coords =
          interface_logical_coordinates(face_mesh, boundary_direction);
      const auto inertial_coords = logical_to_inertial_map(logical_coords);
      const auto inv_jacobian =
          logical_to_inertial_map.inv_jacobian(logical_coords);

      // Get required fields
      const auto background_fields = solution.variables(
          inertial_coords,
          tmpl::list<
              Xcts::Tags::ConformalFactor<DataVector>,
              gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
              gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>,
              gr::Tags::TraceExtrinsicCurvature<DataVector>>{});
      const auto& conformal_factor =
          get<Xcts::Tags::ConformalFactor<DataVector>>(background_fields);
      const auto& inv_spatial_metric =
          get<gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>>(
              background_fields);
      const auto& extrinsic_curvature =
          get<gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>>(
              background_fields);
      const auto& trace_extrinsic_curvature =
          get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(background_fields);

      // Compute the inverse extrinsic curvature
      tnsr::II<DataVector, 3> inv_extrinsic_curvature;
      tenex::evaluate<ti::I, ti::J>(make_not_null(&inv_extrinsic_curvature),
                                    inv_spatial_metric(ti::I, ti::K) *
                                        inv_spatial_metric(ti::J, ti::L) *
                                        extrinsic_curvature(ti::k, ti::l));

      // Compute Euclidean area element
      const auto flat_area_element =
          euclidean_area_element(inv_jacobian, boundary_direction);

      // Compute Euclidean face normal
      auto flat_face_normal = unnormalized_face_normal(
          face_mesh, logical_to_inertial_map, boundary_direction);
      const auto face_normal_magnitude = magnitude(flat_face_normal);
      for (size_t d = 0; d < 3; ++d) {
        flat_face_normal.get(d) /= get(face_normal_magnitude);
      }

      // Evaluate surface integral
      const auto surface_integrand =
          Xcts::adm_linear_momentum_surface_integrand(
              conformal_factor, inv_spatial_metric, inv_extrinsic_curvature,
              trace_extrinsic_curvature);
      const auto contracted_integrand = tenex::evaluate<ti::I>(
          surface_integrand(ti::I, ti::J) * flat_face_normal(ti::j));
      for (int I = 0; I < 3; I++) {
        surface_integral.get(I) += definite_integral(
            contracted_integrand.get(I) * get(flat_area_element), face_mesh);
      }
    }
  }

  // Check result
  auto custom_approx = Approx::custom().epsilon(10. / distance).scale(1.0);
  const double lorentz_factor = 1. / sqrt(1. - square(boost_speed));
  CHECK(get<0>(surface_integral) == custom_approx(0.));
  CHECK(get<1>(surface_integral) == custom_approx(0.));
  CHECK(get<2>(surface_integral) ==
        custom_approx(lorentz_factor * mass * boost_speed));
}

template <typename Solution>
void test_infinite_volume_integral(const double distance, const double mass,
                                   const double horizon_radius,
                                   const double boost_speed,
                                   const Solution& solution) {
  // Set up domain
  const size_t h_refinement = 1;
  const size_t p_refinement = 6;
  const domain::creators::Sphere shell{
      /* inner_radius */ 2 * horizon_radius,
      /* outer_radius */ distance,
      /* interior */ domain::creators::Sphere::Excision{},
      /* initial_refinement */ h_refinement,
      /* initial_number_of_grid_points */ p_refinement + 1,
      /* use_equiangular_map */ true,
      /* equatorial_compression */ {},
      /* radial_partitioning */ {},
      /* radial_distribution */ domain::CoordinateMaps::Distribution::Inverse};
  const auto shell_domain = shell.create_domain();
  const auto& blocks = shell_domain.blocks();
  const auto& initial_ref_levels = shell.initial_refinement_levels();
  const auto element_ids = initial_element_ids(initial_ref_levels);
  const Mesh<3> mesh{p_refinement + 1, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Mesh<2> face_mesh{p_refinement + 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  // Initialize surface integral
  tnsr::I<double, 3> total_integral({0., 0., 0.});

  // Compute integrals by summing over each element
  for (const auto& element_id : element_ids) {
    // Get element information
    const auto& current_block = blocks.at(element_id.block_id());
    const auto current_element = domain::Initialization::create_initial_element(
        element_id, current_block, initial_ref_levels);
    const ElementMap<3, Frame::Inertial> logical_to_inertial_map(
        element_id, current_block.stationary_map().get_clone());

    // Get 3D coordinates
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = logical_to_inertial_map(logical_coords);
    const auto jacobian = logical_to_inertial_map.jacobian(logical_coords);
    const auto det_jacobian = determinant(jacobian);
    const auto inv_jacobian =
        logical_to_inertial_map.inv_jacobian(logical_coords);

    // Get required fields
    const auto solution_fields = solution.variables(
        inertial_coords,
        tmpl::list<
            Xcts::Tags::ConformalFactor<DataVector>,
            ::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>,
            Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
            Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
            gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
            gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>,
            gr::Tags::TraceExtrinsicCurvature<DataVector>,
            Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                       Frame::Inertial>,
            Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                       Frame::Inertial>>{});
    const auto& conformal_factor =
        get<Xcts::Tags::ConformalFactor<DataVector>>(solution_fields);
    const auto& deriv_conformal_factor =
        get<::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>>(solution_fields);
    const auto& conformal_metric =
        get<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
            solution_fields);
    const auto& inv_conformal_metric =
        get<Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
            solution_fields);
    const auto& inv_spatial_metric =
        get<gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>>(
            solution_fields);
    const auto& extrinsic_curvature =
        get<gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>>(
            solution_fields);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(solution_fields);
    const auto& conformal_christoffel_second_kind =
        get<Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                       Frame::Inertial>>(
            solution_fields);
    const auto& conformal_christoffel_contracted =
        get<Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                       Frame::Inertial>>(
            solution_fields);

    // Compute the inverse extrinsic curvature
    tnsr::II<DataVector, 3> inv_extrinsic_curvature;
    tenex::evaluate<ti::I, ti::J>(make_not_null(&inv_extrinsic_curvature),
                                  inv_spatial_metric(ti::I, ti::K) *
                                      inv_spatial_metric(ti::J, ti::L) *
                                      extrinsic_curvature(ti::k, ti::l));

    const auto surface_integrand = Xcts::adm_linear_momentum_surface_integrand(
        conformal_factor, inv_spatial_metric, inv_extrinsic_curvature,
        trace_extrinsic_curvature);

    const auto volume_integrand = Xcts::adm_linear_momentum_volume_integrand(
        surface_integrand, conformal_factor, deriv_conformal_factor,
        conformal_metric, inv_conformal_metric,
        conformal_christoffel_second_kind, conformal_christoffel_contracted);
    for (int I = 0; I < 3; I++) {
      total_integral.get(I) +=
          definite_integral(volume_integrand.get(I) * get(det_jacobian), mesh);
    }

    // Loop over external boundaries
    for (auto boundary_direction : current_element.external_boundaries()) {
      // Skip interfaces not at the outer boundary
      if (boundary_direction != Direction<3>::lower_zeta()) {
        continue;
      }

      // Get interface coordinates.
      const auto face_logical_coords =
          interface_logical_coordinates(face_mesh, boundary_direction);
      const auto face_inv_jacobian =
          logical_to_inertial_map.inv_jacobian(face_logical_coords);

      // Slice required fields to the interface
      const size_t slice_index =
          index_to_slice_at(mesh.extents(), boundary_direction);
      const auto& face_surface_integrand =
          data_on_slice(surface_integrand, mesh.extents(),
                        boundary_direction.dimension(), slice_index);

      // Compute Euclidean area element
      const auto flat_area_element =
          euclidean_area_element(face_inv_jacobian, boundary_direction);

      // Compute Euclidean face normal
      auto flat_face_normal = unnormalized_face_normal(
          face_mesh, logical_to_inertial_map, boundary_direction);
      const auto face_normal_magnitude = magnitude(flat_face_normal);
      for (size_t d = 0; d < 3; ++d) {
        flat_face_normal.get(d) /= get(face_normal_magnitude);
      }

      // Compute surface integrand
      const auto contracted_integrand = tenex::evaluate<ti::I>(
          -face_surface_integrand(ti::I, ti::J) * flat_face_normal(ti::j));

      // Compute contribution to surface integral
      for (int I = 0; I < 3; I++) {
        total_integral.get(I) += definite_integral(
            contracted_integrand.get(I) * get(flat_area_element), face_mesh);
      }
    }
  }

  // Check result
  const double lorentz_factor = 1. / sqrt(1. - square(boost_speed));
  auto custom_approx = Approx::custom().epsilon(10. / distance).scale(1.0);
  CHECK(get<0>(total_integral) == custom_approx(0.));
  CHECK(get<1>(total_integral) == custom_approx(0.));
  CHECK(get<2>(total_integral) ==
        custom_approx(lorentz_factor * mass * boost_speed));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.AdmLinearMomentum",
                  "[Unit][PointwiseFunctions]") {
  {
    INFO("Schwarzschild in Kerr-Schild coordinates");
    const double mass = 1.;
    const double horizon_radius = 2. * mass;
    const double boost_speed = 0.;
    const Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild> solution(
        mass, std::array<double, 3>{{0., 0., 0.}},
        std::array<double, 3>{{0., 0., 0.}},
        std::array<double, 3>{{0., 0., boost_speed}});
    for (const double distance : std::array<double, 3>({1.e4, 1.e5, 1.e6})) {
      test_infinite_surface_integral(distance, mass, horizon_radius,
                                     boost_speed, solution);
      test_infinite_volume_integral(distance, mass, horizon_radius, boost_speed,
                                    solution);
    }
  }
  {
    INFO("Boosted Schwarzschild in Kerr-Schild coordinates");
    const double mass = 1.;
    const double horizon_radius = 2. * mass;
    const double boost_speed = 0.5;
    const Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild> solution(
        mass, std::array<double, 3>{{0., 0., 0.}},
        std::array<double, 3>{{0., 0., 0.}},
        std::array<double, 3>{{0., 0., boost_speed}});
    for (const double distance : std::array<double, 3>({1.e4, 1.e5, 1.e6})) {
      test_infinite_surface_integral(distance, mass, horizon_radius,
                                     boost_speed, solution);
      // Note: the volume integral currently doesn't work with this test case.
    }
  }
  {
    INFO("Schwarzschild in isotropic coordinates");
    const double mass = 1.;
    const double horizon_radius = 0.5 * mass;
    const double boost_speed = 0.;
    const Xcts::Solutions::Schwarzschild solution(
        mass, Xcts::Solutions::SchwarzschildCoordinates::Isotropic);
    for (const double distance : std::array<double, 3>({1.e4, 1.e5, 1.e6})) {
      test_infinite_surface_integral(distance, mass, horizon_radius,
                                     boost_speed, solution);
      test_infinite_volume_integral(distance, mass, horizon_radius, boost_speed,
                                    solution);
    }
  }

  // Test integrands against Python implementation with random values.
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Xcts"};
  const DataVector used_for_size{5};
  pypp::check_with_random_values<1>(
      static_cast<void (*)(
          gsl::not_null<tnsr::II<DataVector, 3>*>, const Scalar<DataVector>&,
          const tnsr::II<DataVector, 3>&, const tnsr::II<DataVector, 3>&,
          const Scalar<DataVector>&)>(
          &Xcts::adm_linear_momentum_surface_integrand),
      "AdmLinearMomentum", {"adm_linear_momentum_surface_integrand"},
      {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<void (*)(
          gsl::not_null<tnsr::I<DataVector, 3>*>,
          const tnsr::II<DataVector, 3>&, const Scalar<DataVector>&,
          const tnsr::i<DataVector, 3>&, const tnsr::ii<DataVector, 3>&,
          const tnsr::II<DataVector, 3>&, const tnsr::Ijj<DataVector, 3>&,
          const tnsr::i<DataVector, 3>&)>(
          &Xcts::adm_linear_momentum_volume_integrand),
      "AdmLinearMomentum", {"adm_linear_momentum_volume_integrand"},
      {{{-1, 1.}}}, used_for_size);
}
