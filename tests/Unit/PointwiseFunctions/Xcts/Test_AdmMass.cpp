// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/AreaElement.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "PointwiseFunctions/Xcts/AdmLinearMomentum.hpp"
#include "PointwiseFunctions/Xcts/AdmMass.hpp"

namespace {

template <typename Solution>
void test_infinite_surface_integral(const double distance, const double mass,
                                    const double horizon_radius,
                                    const double boost_speed,
                                    const Solution& solution) {
  // Set up domain
  const size_t h_refinement = 1;
  const size_t p_refinement = 6;
  const domain::creators::Sphere shell{
      /* inner_radius */ 1.1 * horizon_radius,
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
  Scalar<double> surface_integral(0.);

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

      // Get interface coordinates
      const auto logical_coords =
          interface_logical_coordinates(face_mesh, boundary_direction);
      const auto inertial_coords = logical_to_inertial_map(logical_coords);
      const auto inv_jacobian =
          logical_to_inertial_map.inv_jacobian(logical_coords);

      // Get required fields on the interface
      const auto background_fields = solution.variables(
          inertial_coords,
          tmpl::list<
              ::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                            tmpl::size_t<3>, Frame::Inertial>,
              Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
              Xcts::Tags::InverseConformalMetric<DataVector, 3,
                                                 Frame::Inertial>,
              Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                         Frame::Inertial>,
              Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                         Frame::Inertial>>{});
      const auto& deriv_conformal_factor =
          get<::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                            tmpl::size_t<3>, Frame::Inertial>>(
              background_fields);
      const auto& conformal_metric =
          get<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
              background_fields);
      const auto& inv_conformal_metric = get<
          Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
          background_fields);
      const auto& conformal_christoffel_second_kind =
          get<Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                         Frame::Inertial>>(
              background_fields);
      const auto& conformal_christoffel_contracted =
          get<Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                         Frame::Inertial>>(
              background_fields);

      // Compute conformal area element
      const auto sqrt_det_conformal_metric =
          Scalar<DataVector>(sqrt(get(determinant(conformal_metric))));
      const auto conformal_area_element =
          area_element(inv_jacobian, boundary_direction, inv_conformal_metric,
                       sqrt_det_conformal_metric);

      // Compute conformal face normal
      auto conformal_face_normal = unnormalized_face_normal(
          face_mesh, logical_to_inertial_map, boundary_direction);
      const auto face_normal_magnitude =
          magnitude(conformal_face_normal, inv_conformal_metric);
      for (size_t d = 0; d < 3; ++d) {
        conformal_face_normal.get(d) /= get(face_normal_magnitude);
      }

      // Compute and contract surface integrand
      const auto surface_integrand = Xcts::adm_mass_surface_integrand(
          deriv_conformal_factor, inv_conformal_metric,
          conformal_christoffel_second_kind, conformal_christoffel_contracted);
      const auto contracted_integrand = tenex::evaluate(
          surface_integrand(ti::I) * conformal_face_normal(ti::i));

      // Compute contribution to surface integral
      surface_integral.get() += definite_integral(
          get(contracted_integrand) * get(conformal_area_element), face_mesh);
    }
  }

  // Check result
  auto custom_approx = Approx::custom().epsilon(10. / distance).scale(1.0);
  const double lorentz_factor = 1. / sqrt(1. - square(boost_speed));
  CHECK(get(surface_integral) == custom_approx(lorentz_factor * mass));
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
      /* inner_radius */ 1.1 * horizon_radius,
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

  // Initialize "reduced" integral.
  Scalar<double> total_integral(0.);

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
    const auto background_fields = solution.variables(
        inertial_coords, mesh, inv_jacobian,
        tmpl::list<
            Xcts::Tags::ConformalFactor<DataVector>,
            ::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>,
            Xcts::Tags::ConformalRicciScalar<DataVector>,
            gr::Tags::TraceExtrinsicCurvature<DataVector>,
            Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
                DataVector>,
            Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
            ::Tags::deriv<
                Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                tmpl::size_t<3>, Frame::Inertial>,
            Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
            Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                       Frame::Inertial>,
            ::Tags::deriv<Xcts::Tags::ConformalChristoffelSecondKind<
                              DataVector, 3, Frame::Inertial>,
                          tmpl::size_t<3>, Frame::Inertial>,
            Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                       Frame::Inertial>>{});
    const auto& conformal_factor =
        get<Xcts::Tags::ConformalFactor<DataVector>>(background_fields);
    const auto& deriv_conformal_factor =
        get<::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>>(background_fields);
    const auto& conformal_ricci_scalar =
        get<Xcts::Tags::ConformalRicciScalar<DataVector>>(background_fields);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(background_fields);
    const auto& longitudinal_shift_minus_dt_conformal_metric_over_lapse_square =
        get<Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
            DataVector>>(background_fields);
    const auto& conformal_metric =
        get<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& deriv_conformal_metric = get<::Tags::deriv<
        Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>>(background_fields);
    const auto& inv_conformal_metric =
        get<Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& conformal_christoffel_second_kind =
        get<Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                       Frame::Inertial>>(
            background_fields);
    const auto& deriv_conformal_christoffel_second_kind =
        get<::Tags::deriv<Xcts::Tags::ConformalChristoffelSecondKind<
                              DataVector, 3, Frame::Inertial>,
                          tmpl::size_t<3>, Frame::Inertial>>(background_fields);
    const auto& conformal_christoffel_contracted =
        get<Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                       Frame::Inertial>>(
            background_fields);

    const auto deriv_inv_conformal_metric =
        tenex::evaluate<ti::i, ti::J, ti::K>(
            inv_conformal_metric(ti::J, ti::L) *
                inv_conformal_metric(ti::K, ti::M) *
                (deriv_conformal_metric(ti::i, ti::l, ti::m) -
                 conformal_christoffel_second_kind(ti::N, ti::i, ti::l) *
                     conformal_metric(ti::n, ti::m) -
                 conformal_christoffel_second_kind(ti::N, ti::i, ti::m) *
                     conformal_metric(ti::l, ti::n)) -
            conformal_christoffel_second_kind(ti::J, ti::i, ti::l) *
                inv_conformal_metric(ti::L, ti::K) -
            conformal_christoffel_second_kind(ti::K, ti::i, ti::l) *
                inv_conformal_metric(ti::J, ti::L));

    const auto energy_density =
        make_with_value<Scalar<DataVector>>(inertial_coords, 0.0);

    const auto sqrt_det_conformal_metric =
        Scalar<DataVector>(sqrt(get(determinant(conformal_metric))));

    // Evaluate volume integral.
    const auto volume_integrand = Xcts::adm_mass_volume_integrand(
        conformal_factor, conformal_ricci_scalar, trace_extrinsic_curvature,
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
        energy_density, inv_conformal_metric, deriv_inv_conformal_metric,
        conformal_christoffel_second_kind, conformal_christoffel_contracted,
        deriv_conformal_christoffel_second_kind);
    total_integral.get() += definite_integral(
        get(volume_integrand) * get(sqrt_det_conformal_metric) *
            get(det_jacobian),
        mesh);

    // Loop over external boundaries.
    for (auto boundary_direction : current_element.external_boundaries()) {
      // Skip interfaces not at the inner boundary.
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
      const auto& face_deriv_conformal_factor =
          data_on_slice(deriv_conformal_factor, mesh.extents(),
                        boundary_direction.dimension(), slice_index);
      const auto& face_inv_conformal_metric =
          data_on_slice(inv_conformal_metric, mesh.extents(),
                        boundary_direction.dimension(), slice_index);
      const auto& face_conformal_christoffel_second_kind =
          data_on_slice(conformal_christoffel_second_kind, mesh.extents(),
                        boundary_direction.dimension(), slice_index);
      const auto& face_conformal_christoffel_contracted =
          data_on_slice(conformal_christoffel_contracted, mesh.extents(),
                        boundary_direction.dimension(), slice_index);
      const auto& face_sqrt_det_conformal_metric =
          data_on_slice(sqrt_det_conformal_metric, mesh.extents(),
                        boundary_direction.dimension(), slice_index);

      // Compute conformal area element
      const auto conformal_area_element = area_element(
          face_inv_jacobian, boundary_direction, face_inv_conformal_metric,
          face_sqrt_det_conformal_metric);

      // Compute conformal face normal
      auto conformal_face_normal = unnormalized_face_normal(
          face_mesh, logical_to_inertial_map, boundary_direction);
      const auto face_normal_magnitude =
          magnitude(conformal_face_normal, face_inv_conformal_metric);
      for (size_t d = 0; d < 3; ++d) {
        conformal_face_normal.get(d) /= get(face_normal_magnitude);
      }

      // Evaluate surface integral.
      const auto surface_integrand = Xcts::adm_mass_surface_integrand(
          face_deriv_conformal_factor, face_inv_conformal_metric,
          face_conformal_christoffel_second_kind,
          face_conformal_christoffel_contracted);
      const auto contracted_integrand = tenex::evaluate(
          -surface_integrand(ti::I) * conformal_face_normal(ti::i));

      // Compute contribution to surface integral
      total_integral.get() += definite_integral(
          get(contracted_integrand) * get(conformal_area_element), face_mesh);
    }
  }

  // Check result
  const double lorentz_factor = 1. / sqrt(1. - square(boost_speed));
  auto custom_approx = Approx::custom().epsilon(10. / distance).scale(1.0);
  CHECK(get(total_integral) == custom_approx(lorentz_factor * mass));
}

}  // namespace

// [[TimeOut, 60]]
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.AdmMass",
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
}
