// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeElementFacesGridCoordinates.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/CartesianProduct.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {
void test_excision_sphere_tag() {
  const std::unique_ptr<DomainCreator<3>> shell =
      std::make_unique<domain::creators::Shell>(
          1., 2., 2, std::array<size_t, 2>{{4, 4}}, true);
  const auto shell_excision_sphere =
      shell->create_domain().excision_spheres().at("ExcisionSphere");

  CHECK(Tags::ExcisionSphere<3>::create_from_options(shell, "ExcisionSphere") ==
        shell_excision_sphere);
  CHECK_THROWS_WITH(
      Tags::ExcisionSphere<3>::create_from_options(shell,
                                                   "OtherExcisionSphere"),
      Catch::Matchers::Contains(
          "Specified excision sphere 'OtherExcisionSphere' not available. "
          "Available excision spheres are: (ExcisionSphere)"));
}

void test_compute_centered_face_coordinates() {
  static constexpr size_t Dim = 3;
  ::TestHelpers::db::test_compute_tag<
      Tags::CenteredFaceCoordinatesCompute<Dim>>("CenteredFaceCoordinates");

  for (const auto& [initial_refinement, quadrature] :
       cartesian_product(std::array<size_t, 2>{{0, 1}},
                         make_array(Spectral::Quadrature::Gauss,
                                    Spectral::Quadrature::GaussLobatto))) {
    CAPTURE(initial_refinement);
    CAPTURE(quadrature);

    // we create two shells with different resolutions
    const size_t extents_1 = 5;
    const size_t extents_2 = 7;
    const domain::creators::Shell shell_1{
        1.5, 3., initial_refinement, {extents_1, extents_1}};
    const domain::creators::Shell shell_2{
        1.5, 3., initial_refinement, {extents_2, extents_2}};
    const auto& initial_extents_1 = shell_1.initial_extents();
    const auto& initial_extents_2 = shell_2.initial_extents();

    const auto domain = shell_1.create_domain();
    const auto& blocks = domain.blocks();
    const auto& excision_sphere =
        domain.excision_spheres().at("ExcisionSphere");
    const auto& initial_refinements = shell_1.initial_refinement_levels();
    const auto element_ids = initial_element_ids(initial_refinements);

    // this worldtube tag should hold the same coordinates of all elements in a
    // map so we use it to check
    std::unordered_map<ElementId<Dim>, tnsr::I<DataVector, Dim, Frame::Grid>>
        all_faces_grid_coords_1{};
    std::unordered_map<ElementId<Dim>, tnsr::I<DataVector, Dim, Frame::Grid>>
        all_faces_grid_coords_2{};
    Initialization::InitializeElementFacesGridCoordinates<Dim>::apply(
        make_not_null(&all_faces_grid_coords_1), domain, initial_extents_1,
        initial_refinements, quadrature, excision_sphere);
    Initialization::InitializeElementFacesGridCoordinates<Dim>::apply(
        make_not_null(&all_faces_grid_coords_2), domain, initial_extents_2,
        initial_refinements, quadrature, excision_sphere);

    for (const auto& element_id : element_ids) {
      const auto& my_block = blocks.at(element_id.block_id());
      const auto element = domain::Initialization::create_initial_element(
          element_id, my_block, initial_refinements);
      const ElementMap element_map(
          element_id, my_block.stationary_map().get_to_grid_frame());
      const auto mesh_1 = domain::Initialization::create_initial_mesh(
          initial_extents_1, element_id, quadrature);
      const auto grid_coords_1 = element_map(logical_coordinates(mesh_1));
      const auto mesh_2 = domain::Initialization::create_initial_mesh(
          initial_extents_2, element_id, quadrature);
      const auto grid_coords_2 = element_map(logical_coordinates(mesh_2));

      auto box = db::create<
          db::AddSimpleTags<Tags::ExcisionSphere<Dim>,
                            domain::Tags::Element<Dim>,
                            domain::Tags::Coordinates<Dim, Frame::Grid>,
                            domain::Tags::Mesh<Dim>>,
          db::AddComputeTags<Tags::CenteredFaceCoordinatesCompute<Dim>>>(
          excision_sphere, element, grid_coords_1, mesh_1);

      const auto centered_face_coordinates_1 =
          db::get<Tags::CenteredFaceCoordinates<Dim>>(box);
      if (all_faces_grid_coords_1.count(element_id)) {
        CHECK(centered_face_coordinates_1.has_value());
        CHECK_ITERABLE_APPROX(centered_face_coordinates_1.value(),
                              all_faces_grid_coords_1.at(element_id));
      } else {
        CHECK(not centered_face_coordinates_1.has_value());
      }

      // we mutate to the differently refined grid, simulating p-refinement
      ::Initialization::mutate_assign<
          tmpl::list<domain::Tags::Mesh<Dim>,
                     domain::Tags::Coordinates<Dim, Frame::Grid>>>(
          make_not_null(&box), mesh_2, grid_coords_2);

      const auto centered_face_coordinates_2 =
          db::get<Tags::CenteredFaceCoordinates<Dim>>(box);
      if (all_faces_grid_coords_2.count(element_id)) {
        // p-refinement should not change which elements are abutting i.e. have
        // a value
        CHECK(centered_face_coordinates_1.has_value());
        CHECK(centered_face_coordinates_2.has_value());
        CHECK_ITERABLE_APPROX(centered_face_coordinates_2.value(),
                              all_faces_grid_coords_2.at(element_id));
      } else {
        CHECK(not centered_face_coordinates_1.has_value());
        CHECK(not centered_face_coordinates_2.has_value());
      }
    }
  }
}

void test_inertial_particle_position_compute() {
  static constexpr size_t Dim = 3;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-10., 10.);
  const tnsr::I<double, Dim, Frame::Grid> grid_coords_center{
      {dist(gen), dist(gen), 0.}};
  const double orbit_radius = get(magnitude(grid_coords_center));
  const double angular_velocity = 1. / (sqrt(orbit_radius) * orbit_radius);
  const ::ExcisionSphere<Dim> excision_sphere{2., grid_coords_center, {}};
  const double initial_time = 0.;
  auto box = db::create<
      db::AddSimpleTags<Tags::ExcisionSphere<Dim>, ::Tags::Time>,
      db::AddComputeTags<Tags::InertialParticlePositionCompute<Dim>>>(
      excision_sphere, initial_time);
  for (size_t i = 0; i < 100; ++i) {
    const double random_time = dist(gen);
    ::Initialization::mutate_assign<tmpl::list<::Tags::Time>>(
        make_not_null(&box), random_time);
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
            domain::CoordinateMaps::Rotation<Dim>(
                angular_velocity * random_time, 0., 0.));
    const auto& inertial_pos =
        db::get<Tags::InertialParticlePosition<Dim>>(box);
    CHECK_ITERABLE_APPROX(inertial_pos, coordinate_map(grid_coords_center));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.Worldtube.Tags",
                  "[Unit][Evolution]") {
  CHECK(TestHelpers::test_option_tag<OptionTags::ExcisionSphere>("Worldtube") ==
        "Worldtube");
  TestHelpers::db::test_simple_tag<Tags::ExcisionSphere<3>>("ExcisionSphere");
  TestHelpers::db::test_simple_tag<Tags::ElementFacesGridCoordinates<3>>(
      "ElementFacesGridCoordinates");
  TestHelpers::db::test_simple_tag<Tags::CenteredFaceCoordinates<3>>(
      "CenteredFaceCoordinates");
  TestHelpers::db::test_simple_tag<Tags::InertialParticlePosition<3>>(
      "InertialParticlePosition");

  test_excision_sphere_tag();
  test_compute_centered_face_coordinates();
  test_inertial_particle_position_compute();
}
}  // namespace CurvedScalarWave::Worldtube
