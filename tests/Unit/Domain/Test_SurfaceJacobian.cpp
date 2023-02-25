// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/SurfaceJacobian.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {
namespace {

// Checks the euclidean surface element against the magnitude of the
// unnormalized normal vector.
template <typename TargetFrame>
void test_surface_element_against_normal_vector() {
  const auto element_map_3d = ElementMap<3, TargetFrame>(
      ElementId<3>(0),
      make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
          CoordinateMaps::Wedge(1., 2., 1., 1., OrientationMap<3>{}, true)));

  const Mesh<2> interface_mesh{8, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};
  const Direction<3> direction{};
  const auto coords = interface_logical_coordinates(interface_mesh, direction);
  const auto inverse_jacobian = element_map_3d.inv_jacobian(coords);
  const auto det_volume = determinant(inverse_jacobian);
  const auto result_1 =
      euclidean_surface_jacobian(inverse_jacobian, det_volume, direction);
  const auto result_2 = euclidean_surface_jacobian(inverse_jacobian, direction);
  const auto face_normal =
      unnormalized_face_normal(interface_mesh, element_map_3d, direction);
  const auto face_normal_mag = magnitude(face_normal);
  const DataVector face_normal_mag_corrected =
      get(face_normal_mag) / get(det_volume);
  CHECK_ITERABLE_APPROX(get(result_1), get(result_2));
  CHECK_ITERABLE_APPROX(face_normal_mag_corrected, get(result_1));
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
        const auto det = euclidean_surface_jacobian(
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
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.SurfaceElement", "[Domain][Unit]") {
  test_surface_element_against_normal_vector<Frame::Grid>();
  test_surface_element_against_normal_vector<Frame::Inertial>();
  test_sphere_integral();
}

}  // namespace domain
