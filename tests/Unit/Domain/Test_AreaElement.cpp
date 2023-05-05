// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/AreaElement.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/Composition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/KerrHorizonConforming.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {
namespace {

// Checks the euclidean and curved area element against the magnitude of the
// unnormalized normal vector.
void test_area_element_against_normal_vector() {
  using TargetFrame = Frame::Inertial;
  using inv_spatial_metric_tag =
      gr::Tags::InverseSpatialMetric<DataVector, 3, TargetFrame>;
  using sqrt_det_spatial_metric_tag =
      gr::Tags::SqrtDetSpatialMetric<DataVector>;
  const auto element_map_3d = ElementMap<3, TargetFrame>(
      ElementId<3>(0),
      make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
          CoordinateMaps::Wedge(1., 2., 1., 1., OrientationMap<3>{}, true)));

  gr::Solutions::Minkowski<3> minkowski{};

  const Mesh<2> interface_mesh{8, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};
  const Direction<3> direction{};
  const auto logical_coords =
      interface_logical_coordinates(interface_mesh, direction);
  const auto inverse_jacobian = element_map_3d.inv_jacobian(logical_coords);
  const auto det_volume = determinant(inverse_jacobian);
  const auto face_normal =
      unnormalized_face_normal(interface_mesh, element_map_3d, direction);
  const auto inertial_coords = element_map_3d(logical_coords);

  // test euclidean area element
  const auto result_1_euclidean =
      euclidean_area_element(inverse_jacobian, det_volume, direction);
  const auto result_2_euclidean =
      euclidean_area_element(inverse_jacobian, direction);
  const auto face_normal_mag_euclidean = magnitude(face_normal);
  const DataVector face_normal_mag_corrected_euclidean =
      get(face_normal_mag_euclidean) / get(det_volume);
  CHECK_ITERABLE_APPROX(get(result_1_euclidean), get(result_2_euclidean));
  CHECK_ITERABLE_APPROX(face_normal_mag_corrected_euclidean,
                        get(result_1_euclidean));

  // test curved area element: Minkowski
  const auto minkowski_vars = minkowski.variables(
      inertial_coords, 0.,
      tmpl::list<inv_spatial_metric_tag, sqrt_det_spatial_metric_tag>{});
  const auto inv_spatial_metric_minkowski =
      get<inv_spatial_metric_tag>(minkowski_vars);
  const auto sqrt_det_spatial_metric =
      get<sqrt_det_spatial_metric_tag>(minkowski_vars);
  const auto result_1_minkowski =
      area_element(inverse_jacobian, det_volume, direction,
                   inv_spatial_metric_minkowski, sqrt_det_spatial_metric);
  const auto result_2_minkowski =
      area_element(inverse_jacobian, direction, inv_spatial_metric_minkowski,
                   sqrt_det_spatial_metric);
  CHECK_ITERABLE_APPROX(get(result_1_minkowski), get(result_2_minkowski));
  CHECK_ITERABLE_APPROX(face_normal_mag_corrected_euclidean,
                        get(result_1_minkowski));
}

// A test function sin^2(theta) * cos^2(phi) which evaluates to 4/3 * pi * R^2
// when integrated over a sphere with radius R.
DataVector test_function(const tnsr::I<DataVector, 3>& inertial_coords) {
  const DataVector theta =
      atan2(hypot(get<0>(inertial_coords), get<1>(inertial_coords)),
            get<2>(inertial_coords));
  const DataVector phi =
      atan2(get<1>(inertial_coords), get<0>(inertial_coords));
  return sin(theta) * sin(theta) * cos(phi) * cos(phi);
}

// Integrates the area and a test function over the excision surface of a shell
void test_sphere_integral() {
  for (const auto& [inner_radius, ref_level] : cartesian_product(
           make_array(1e-4, 1e-2, 1.4), std::array<size_t, 3>{{0, 1, 2}})) {
    domain::creators::Sphere shell{
        inner_radius, 3.,    domain::creators::Sphere::Excision{},
        ref_level,    10_st, true};
    const auto shell_domain = shell.create_domain();
    const auto& blocks = shell_domain.blocks();
    const auto& initial_ref_levels = shell.initial_refinement_levels();
    const auto element_ids = initial_element_ids(initial_ref_levels);
    const auto excision_sphere =
        shell_domain.excision_spheres().at("ExcisionSphere");
    const Mesh<2> face_mesh{10, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
    double sphere_surface = 0.;
    double test_func_integral = 0.;
    for (const auto& element_id : element_ids) {
      const auto& current_block = blocks.at(element_id.block_id());
      const auto element_abutting_direction =
          excision_sphere.abutting_direction(element_id);
      if (element_abutting_direction.has_value()) {
        const auto face_logical_coords = interface_logical_coordinates(
            face_mesh, element_abutting_direction.value());
        const ElementMap logical_to_grid_map(
            element_id, current_block.stationary_map().get_clone());
        const auto inv_jacobian =
            logical_to_grid_map.inv_jacobian(face_logical_coords);
        const auto inertial_coords = logical_to_grid_map(face_logical_coords);
        const auto det_jacobian = ::determinant(inv_jacobian);
        const auto det = euclidean_area_element(
            inv_jacobian, det_jacobian, element_abutting_direction.value());
        sphere_surface += definite_integral(get(det), face_mesh);
        test_func_integral += definite_integral(
            test_function(inertial_coords) * get(det), face_mesh);
      }
    }
    auto custom_approx = Approx::custom().epsilon(1e-10);
    CHECK(sphere_surface ==
          custom_approx(4. * M_PI * inner_radius * inner_radius));
    CHECK(test_func_integral ==
          custom_approx(4. / 3. * M_PI * inner_radius * inner_radius));
  }
}

// checks that the integral over the horizon of a Kerr black hole
// corresponds to the expected area
void test_kerr_area() {
  for (const auto& [ref_level, mass, spin] : cartesian_product(
           std::array<size_t, 3>{{0, 1}}, make_array(0.001, 1., 100.),
           std::array<std::array<double, 3>, 3>{{make_array(0., 0., 0.),
                                                 make_array(0.1, 0.2, 0.3),
                                                 make_array(0., 0., 0.99)}})) {
    const double horizon_radius = mass * (1 + sqrt(1 - dot(spin, spin)));
    domain::creators::Sphere shell{horizon_radius,
                                   10. * horizon_radius,
                                   domain::creators::Sphere::Excision{},
                                   ref_level,
                                   size_t(10),
                                   true};
    const auto shell_domain = shell.create_domain();
    const auto& blocks = shell_domain.blocks();
    const auto& initial_ref_levels = shell.initial_refinement_levels();
    const auto element_ids = initial_element_ids(initial_ref_levels);
    const auto excision_sphere =
        shell_domain.excision_spheres().at("ExcisionSphere");
    const Mesh<2> face_mesh{10, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
    // Shell does not distort to conform to the horizon in Kerr-Schild
    // coordinates for spinning black holes, so we append the horizon
    // conforming map
    const auto kerr_conforming_map =
        CoordinateMap<Frame::Grid, Frame::Inertial,
                      domain::CoordinateMaps::KerrHorizonConforming>(
            domain::CoordinateMaps::KerrHorizonConforming{mass, spin});
    gr::Solutions::KerrSchild kerr_schild{mass, spin, {0., 0., 0.}};
    double horizon_area = 0.;
    for (const auto& element_id : element_ids) {
      const auto& current_block = blocks.at(element_id.block_id());
      const auto element_abutting_direction =
          excision_sphere.abutting_direction(element_id);
      if (element_abutting_direction.has_value()) {
        const auto face_logical_coords = interface_logical_coordinates(
            face_mesh, element_abutting_direction.value());
        const domain::CoordinateMaps::Composition logical_to_inertial_map{
            domain::element_to_block_logical_map(element_id),
            current_block.stationary_map().get_to_grid_frame(),
            kerr_conforming_map.get_clone()};
        const auto inverse_jacobian =
            logical_to_inertial_map.inv_jacobian(face_logical_coords);
        const auto spacetime_vars = kerr_schild.variables(
            logical_to_inertial_map(face_logical_coords), 0.,
            tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3>,
                       gr::Tags::SqrtDetSpatialMetric<DataVector>>{});
        const auto curved_area_element = area_element(
            inverse_jacobian, element_abutting_direction.value(),
            get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(spacetime_vars),
            get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(spacetime_vars));
        horizon_area += definite_integral(get(curved_area_element), face_mesh);
      }
    }
    auto custom_approx = Approx::custom().epsilon(1e-8);
    CHECK(horizon_area == custom_approx(8. * M_PI * mass * horizon_radius));
  }
}

SPECTRE_TEST_CASE("Unit.Domain.AreaElement", "[Domain][Unit]") {
  test_area_element_against_normal_vector();
  test_sphere_integral();
  test_kerr_area();
}

}  // namespace
}  // namespace domain
