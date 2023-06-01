// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "ParallelAlgorithms/SurfaceFinder/SurfaceFinder.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"

namespace {
// Builds a Tensor out of the ray directions and outputted zeta coordinates.
// Only used when we expect a root to be found along all rays.
tnsr::I<DataVector, 3, Frame::ElementLogical> result_helper(
    const tnsr::I<DataVector, 2, Frame::ElementLogical>& ray_directions,
    const std::vector<std::optional<double>>& zeta_raw) {
  DataVector zeta(zeta_raw.size());
  for (size_t i = 0; i < zeta_raw.size(); i++) {
    zeta[i] = zeta_raw[i].value();
  }
  tnsr::I<DataVector, 3, Frame::ElementLogical> result{
      {{get<0>(ray_directions), get<1>(ray_directions), zeta}}};
  return result;
}

void test_bulging_surface(
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 2, Frame::ElementLogical>& ray_directions,
    const ElementMap<3, Frame::Inertial>& element_map) {
  INFO(
      "Find the level-surface of a Gaussian offset from the origin, such that "
      "the surface bulges in and out of the element. This tests if the "
      "function can find a surface partially in an element.");

  const MathFunctions::Gaussian<3, Frame::Inertial> offset_gaussian{
      10., 5., std::array<double, 3>{0., 0., -1.}};
  const auto data = offset_gaussian(inertial_coords);
  const double target = 2.5;
  const auto radius_zeta =
      SurfaceFinder::find_radial_surface(data, target, mesh, ray_directions);

  const tnsr::I<double, 3, Frame::ElementLogical> result{
      {{get<0>(ray_directions)[1], get<1>(ray_directions)[1],
        radius_zeta[1].value()}}};
  const auto inertial_contour = element_map(result);
  const auto radius_values = get(magnitude(inertial_contour));

  Approx custom_approx = Approx::custom().epsilon(1.e-5).scale(1.0);
  CHECK(radius_zeta[0].has_value() == false);
  CHECK(radius_zeta[1].has_value() == true);
  CHECK(radius_values == custom_approx(sqrt(-25. * log(0.25)) - 1.));
  CHECK(radius_zeta[2].has_value() == false);
}

void test_radius_contour(
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 2, Frame::ElementLogical>& ray_directions,
    const ElementMap<3, Frame::Inertial>& element_map) {
  INFO("Find certain radius contour.");

  const auto data = magnitude(inertial_coords);
  const double target = 4.0;

  const auto radius_zeta =
      SurfaceFinder::find_radial_surface(data, target, mesh, ray_directions);
  const auto result = result_helper(ray_directions, radius_zeta);
  const auto inertial_contour = element_map(result);
  const auto radius_values = get(magnitude(inertial_contour));

  Approx custom_approx = Approx::custom().epsilon(1.e-6).scale(1.0);
  for (size_t i = 0; i < radius_zeta.size(); i++) {
    CHECK(radius_zeta[i].has_value());
    CHECK(radius_values[i] == custom_approx(target));
  }
}

void test_strahlkorper_input(
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const Mesh<3>& mesh, const Domain<3>& domain, const ElementId<3>& id,
    const ElementMap<3, Frame::Inertial>& element_map) {
  INFO(
      "Take the collocation points from a Strahlkorper, find a particular "
      "radius, and compare to the physical radius from the Strahlkorper.");

  const double target = 4.3;
  const auto data = magnitude(inertial_coords);

  const Strahlkorper<Frame::Inertial> strahlkorper{
      5, 5, 4.5, std::array<double, 3>{0., 0., 0.}};
  const auto& ylm = strahlkorper.ylm_spherepack();
  const auto& [theta, phi] = ylm.theta_phi_points();
  const auto x = target * sin(theta) * cos(phi);
  const auto y = target * sin(theta) * sin(phi);
  const auto z = target * cos(theta);
  const tnsr::I<DataVector, 3, Frame::Inertial> points{{{x, y, z}}};
  const std::vector<ElementId<3>> id_vector{id};
  const auto block_coords = block_logical_coordinates(domain, points);
  const auto element_coords =
      element_logical_coordinates(id_vector, block_coords);
  const tnsr::I<DataVector, 2, Frame::ElementLogical> ray_directions{
      {{get<0>(element_coords.at(id).element_logical_coords),
        get<1>(element_coords.at(id).element_logical_coords)}}};

  const auto radius_zeta =
      SurfaceFinder::find_radial_surface(data, target, mesh, ray_directions);
  const auto result = result_helper(ray_directions, radius_zeta);
  const auto inertial_contour = element_map(result);
  const auto radius_values = get(magnitude(inertial_contour));

  Approx custom_approx = Approx::custom().epsilon(1.e-6).scale(1.0);
  for (size_t i = 0; i < radius_zeta.size(); i++) {
    CHECK(radius_zeta[i].has_value());
    CHECK(radius_values[i] == custom_approx(target));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.SurfaceFinder.SurfaceFinder",
                  "[Unit][ParallelAlgorithms]") {
  // Build wedge for tests.
  static constexpr size_t dim = 3;
  const domain::creators::Sphere sphere{
      1., 5., domain::creators::Sphere::InnerCube{0.}, 0_st, 12_st, false};
  const auto domain = sphere.create_domain();
  const auto refinement_levels = sphere.initial_refinement_levels();
  const auto extents = sphere.initial_extents();
  const ElementId<dim> id{0};
  const auto quadrature = Spectral::Quadrature::GaussLobatto;
  const auto& block = domain.blocks()[id.block_id()];
  const auto mesh =
      domain::Initialization::create_initial_mesh(extents, id, quadrature);
  const ElementMap<dim, Frame::Inertial> element_map{
      id, block.stationary_map().get_clone()};
  const auto logical_coords = logical_coordinates(mesh);
  const auto inertial_coords = element_map(logical_coords);

  // Ray directions used for some of the tests
  const tnsr::I<DataVector, 2, Frame::ElementLogical> ray_directions{
      {{{-1., 0., 1.}, {0., 0., 0.}}}};

  test_bulging_surface(inertial_coords, mesh, ray_directions, element_map);
  test_radius_contour(inertial_coords, mesh, ray_directions, element_map);
  test_strahlkorper_input(inertial_coords, mesh, domain, id, element_map);
}
