// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/AreaElement.hpp"
#include "Domain/CoordinateMaps/Composition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/KerrHorizonConforming.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "PointwiseFunctions/Xcts/AdmLinearMomentum.hpp"
#include "PointwiseFunctions/Xcts/AdmMass.hpp"

namespace {

using KerrSchild = Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild>;

void test_mass_surface_integral(const double distance,
                                const size_t refinement_level,
                                const size_t polynomial_order) {
  // Define black hole parameters
  const double mass = 1;
  const double boost_speed = 0.5;
  const double lorentz_factor = 1. / sqrt(1. - square(boost_speed));

  // Get Kerr-Schild solution
  const std::array<double, 3> dimensionless_spin{{0., 0., 0.}};
  const std::array<double, 3> center{{0., 0., 0.}};
  const std::array<double, 3> boost_velocity{{0., 0., boost_speed}};
  const KerrSchild solution{mass, dimensionless_spin, center, boost_velocity};

  // Set up domain
  domain::creators::Sphere shell{
      /* inner_radius */ 2 * mass,
      /* outer_radius */ distance,
      /* interior */ domain::creators::Sphere::Excision{},
      refinement_level,
      polynomial_order + 1,
      true,
      {},
      {},
      domain::CoordinateMaps::Distribution::Logarithmic};
  const auto shell_domain = shell.create_domain();
  const auto& blocks = shell_domain.blocks();
  const auto& initial_ref_levels = shell.initial_refinement_levels();
  const auto element_ids = initial_element_ids(initial_ref_levels);
  const Mesh<2> face_mesh{polynomial_order + 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<3> mesh_3d{polynomial_order + 1, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  // Initialize surface integral
  double surface_integral = 0;

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
      surface_integral += definite_integral(
          get(contracted_integrand) * get(conformal_area_element), face_mesh);
    }
  }

  // Check result
  auto custom_approx = Approx::custom().epsilon(10. / distance).scale(1.0);
  CHECK(surface_integral == custom_approx(lorentz_factor * mass));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.AdmMass",
                  "[Unit][PointwiseFunctions]") {
  const size_t P = 6;
  const size_t L = 1;
  for (double distance : std::array<double, 3>{{1.e3, 1.e5, 1.e10}}) {
    test_mass_surface_integral(distance, L, P);
  }
}
