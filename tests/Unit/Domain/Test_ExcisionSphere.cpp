// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

namespace {
template <size_t VolumeDim>
void check_excision_sphere_work(
    const double radius, const tnsr::I<double, VolumeDim, Frame::Grid> center,
    const std::unordered_map<size_t, Direction<VolumeDim>>&
        abutting_directions) {
  const ExcisionSphere<VolumeDim> excision_sphere(radius, center,
                                                  abutting_directions);

  CHECK_FALSE(excision_sphere.is_time_dependent());
  CHECK(excision_sphere.radius() == radius);
  CHECK(excision_sphere.center() == center);
  CHECK(excision_sphere.abutting_directions() == abutting_directions);
  CHECK(excision_sphere == excision_sphere);
  CHECK_FALSE(excision_sphere != excision_sphere);

  const double diff_radius = 0.001;
  const ExcisionSphere<VolumeDim> excision_sphere_diff_radius(
      diff_radius, center, abutting_directions);
  CHECK(excision_sphere != excision_sphere_diff_radius);
  CHECK_FALSE(excision_sphere == excision_sphere_diff_radius);

  CHECK(get_output(excision_sphere) ==
        "ExcisionSphere:\n"
        "  Radius: " +
            get_output(excision_sphere.radius()) +
            "\n"
            "  Center: " +
            get_output(excision_sphere.center()) +
            "\n"
            "  Abutting directions: " +
            get_output(excision_sphere.abutting_directions()) + "\n");

  test_serialization(excision_sphere);
}

void check_excision_sphere_1d() {
  const double radius = 1.2;
  const tnsr::I<double, 1, Frame::Grid> center{{5.4}};
  check_excision_sphere_work<1>(
      radius, center,
      {{0, Direction<1>::lower_xi()}, {2, Direction<1>::upper_xi()}});
}

void check_excision_sphere_2d() {
  const double radius = 4.2;
  const tnsr::I<double, 2, Frame::Grid> center{{5.4, -2.3}};
  check_excision_sphere_work<2>(
      radius, center,
      {{0, Direction<2>::lower_eta()}, {2, Direction<2>::upper_xi()}});
}

void check_excision_sphere_3d() {
  const double radius = 5.2;
  const tnsr::I<double, 3, Frame::Grid> center{{5.4, -2.3, 9.0}};
  check_excision_sphere_work<3>(
      radius, center,
      {{0, Direction<3>::lower_xi()}, {2, Direction<3>::upper_zeta()}});
}

void test_abutting_direction_shell() {
  const double inner_radius = 1.3;
  for (const auto& ref_level : std::array<size_t, 3>{{0, 1, 2}}) {
    // shell without radial partition (blocks have 2 outer boundaries)
    domain::creators::Sphere shell_plain{
        inner_radius, 3.,   domain::creators::Sphere::Excision{},
        ref_level,    4_st, true};
    // shell with radial partition (blocks have 1 outer boundary)
    domain::creators::Sphere shell_partitioned{
        inner_radius,
        3.,
        domain::creators::Sphere::Excision{},
        ref_level,
        4_st,
        true,
        std::nullopt,
        {2.},
        {std::vector{{domain::CoordinateMaps::Distribution::Linear,
                      domain::CoordinateMaps::Distribution::Linear}}}};
    std::vector<domain::creators::Sphere> shells{};
    shells.push_back(std::move(shell_plain));
    shells.push_back(std::move(shell_partitioned));
    for (const auto& shell : shells) {
      const auto shell_domain = shell.create_domain();
      const auto& blocks = shell_domain.blocks();
      const auto& initial_ref_levels = shell.initial_refinement_levels();
      const auto element_ids = initial_element_ids(initial_ref_levels);
      const auto excision_sphere =
          shell_domain.excision_spheres().at("ExcisionSphere");
      const auto& abutting_directions = excision_sphere.abutting_directions();
      size_t num_excision_neighbors = 0;
      const Mesh<2> face_mesh{10, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
      for (const auto& element_id : element_ids) {
        const auto& current_block = blocks.at(element_id.block_id());
        const auto element_abutting_direction =
            excision_sphere.abutting_direction(element_id);
        if (element_abutting_direction.has_value()) {
          CHECK(abutting_directions.count(element_id.block_id()));
          CHECK(element_abutting_direction.value() ==
                abutting_directions.at(element_id.block_id()));
          const auto face_logical_coords = interface_logical_coordinates(
              face_mesh, element_abutting_direction.value());
          const auto logical_to_grid_map = ElementMap(
              element_id, current_block.stationary_map().get_clone());
          const auto grid_coords = logical_to_grid_map(face_logical_coords);
          const auto radii = magnitude(grid_coords);
          CHECK_ITERABLE_APPROX(radii.get(),
                                DataVector(radii.get().size(), inner_radius));
          num_excision_neighbors++;
        }
      }
      CHECK(num_excision_neighbors == 6 * pow(4, ref_level));
    }
  }
}

void test_error() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ExcisionSphere<3>(-2.0, tnsr::I<double, 3, Frame::Grid>{{3.4, 1.2, -0.9}},
                        {}),
      Catch::Matchers::ContainsSubstring(
          "The ExcisionSphere must have a radius greater than zero."));
#endif
}

void test_time_dependent_maps() {
  using TranslationMap = domain::CoordinateMaps::TimeDependent::Translation<3>;
  PUPable_reg(SINGLE_ARG(
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap>));
  std::unique_ptr<domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>
      time_dep_map = std::make_unique<
          domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap>>(
          TranslationMap{"DuckDuckGoose"});

  ExcisionSphere<3> excision_sphere{
      1.0, tnsr::I<double, 3, Frame::Grid>{{0.0, 0.0, 0.0}}, {}};

  CHECK_FALSE(excision_sphere.is_time_dependent());
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(excision_sphere.moving_mesh_grid_to_inertial_map(),
                    Catch::Matchers::ContainsSubstring(
                        "Trying to access grid to inertial map "
                        "from ExcisionSphere, but it has "
                        "not been set."));
#endif

  excision_sphere.inject_time_dependent_maps(time_dep_map->get_clone());
  test_serialization(excision_sphere);
  CHECK(excision_sphere.is_time_dependent());
  CHECK(excision_sphere.moving_mesh_grid_to_inertial_map() == *time_dep_map);
  test_copy_semantics(excision_sphere);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.ExcisionSphere", "[Domain][Unit]") {
  check_excision_sphere_1d();
  check_excision_sphere_2d();
  check_excision_sphere_3d();
  test_abutting_direction_shell();
  test_error();
  test_time_dependent_maps();
}
