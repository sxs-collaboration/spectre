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
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/InitializeElementFacesGridCoordinates.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Helpers/Evolution/Systems/CurvedScalarWave/Worldtube/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {
void test_excision_sphere_tag() {
  const std::unique_ptr<DomainCreator<3>> shell =
      std::make_unique<domain::creators::Sphere>(
          1., 2., domain::creators::Sphere::Excision{}, 2_st, 4_st, true);
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

void test_compute_face_coordinates_grid() {
  static constexpr size_t Dim = 3;
  ::TestHelpers::db::test_compute_tag<
      Tags::FaceCoordinatesCompute<Dim, Frame::Grid, true>>("FaceCoordinates");
  ::TestHelpers::db::test_compute_tag<
      Tags::FaceCoordinatesCompute<Dim, Frame::Grid, false>>("FaceCoordinates");
  ::TestHelpers::db::test_compute_tag<
      Tags::FaceCoordinatesCompute<Dim, Frame::Inertial, true>>(
      "FaceCoordinates");
  ::TestHelpers::db::test_compute_tag<
      Tags::FaceCoordinatesCompute<Dim, Frame::Inertial, false>>(
      "FaceCoordinates");

  for (const auto& initial_refinement : std::array<size_t, 2>{{0, 1}}) {
    CAPTURE(initial_refinement);
    const auto quadrature = Spectral::Quadrature::GaussLobatto;
    // we create two shells with different resolutions
    const size_t extents_1 = 5;
    const size_t extents_2 = 7;
    const domain::creators::Sphere shell_1{1.5,
                                           3.,
                                           domain::creators::Sphere::Excision{},
                                           initial_refinement,
                                           extents_1,
                                           true};
    const domain::creators::Sphere shell_2{1.5,
                                           3.,
                                           domain::creators::Sphere::Excision{},
                                           initial_refinement,
                                           extents_2,
                                           true};
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
        make_not_null(&all_faces_grid_coords_1), initial_extents_1,
        initial_refinements, quadrature, domain, excision_sphere);
    Initialization::InitializeElementFacesGridCoordinates<Dim>::apply(
        make_not_null(&all_faces_grid_coords_2), initial_extents_2,
        initial_refinements, quadrature, domain, excision_sphere);

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

      auto box =
          db::create<db::AddSimpleTags<
                         Tags::ExcisionSphere<Dim>, domain::Tags::Element<Dim>,
                         domain::Tags::Coordinates<Dim, Frame::Grid>,
                         domain::Tags::Mesh<Dim>>,
                     db::AddComputeTags<
                         Tags::FaceCoordinatesCompute<Dim, Frame::Grid, true>>>(
              excision_sphere, element, grid_coords_1, mesh_1);

      const auto centered_face_coordinates_1 =
          db::get<Tags::FaceCoordinates<Dim, Frame::Grid, true>>(box);
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
          db::get<Tags::FaceCoordinates<Dim, Frame::Grid, true>>(box);
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

void test_compute_face_coordinates() {
  static constexpr size_t Dim = 3;
  const auto domain_creator =
      TestHelpers::CurvedScalarWave::Worldtube::worldtube_binary_compact_object(
          7., 0.2, pow(7, -1.5));
  const double initial_time = 0.;
  auto domain = domain_creator->create_domain();
  const auto excision_sphere = domain.excision_spheres().at("ExcisionSphereA");
  const auto& blocks = domain.blocks();
  const auto initial_extents = domain_creator->initial_extents();
  const auto initial_refinements = domain_creator->initial_refinement_levels();
  const auto element_ids = initial_element_ids(initial_refinements);
  const auto quadrature = Spectral::Quadrature::GaussLobatto;
  std::unordered_map<ElementId<Dim>, tnsr::I<DataVector, Dim, Frame::Grid>>
      all_faces_grid_coords{};
  Initialization::InitializeElementFacesGridCoordinates<Dim>::apply(
      make_not_null(&all_faces_grid_coords), initial_extents,
      initial_refinements, quadrature, domain, excision_sphere);
  for (const auto& element_id : element_ids) {
    const auto& my_block = blocks.at(element_id.block_id());
    const auto element = domain::Initialization::create_initial_element(
        element_id, my_block, initial_refinements);
    const auto mesh = domain::Initialization::create_initial_mesh(
        initial_extents, element_id, quadrature);
    const ElementMap element_map(
        element_id,
        my_block.moving_mesh_logical_to_grid_map().get_to_grid_frame());
    const auto grid_coords = element_map(logical_coordinates(mesh));

    auto box = db::create<
        db::AddSimpleTags<::Tags::Time, Tags::ExcisionSphere<3>,
                          domain::Tags::FunctionsOfTimeInitialize,
                          domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                          domain::Tags::Coordinates<Dim, Frame::Grid>,
                          ::domain::CoordinateMaps::Tags::CoordinateMap<
                              Dim, Frame::Grid, Frame::Inertial>>,
        db::AddComputeTags<
            Tags::InertialParticlePositionCompute<3>,
            ::domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
                ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                              Frame::Inertial>>,
            ::domain::Tags::InertialFromGridCoordinatesCompute<Dim>,
            Tags::FaceCoordinatesCompute<3, Frame::Grid, true>,
            Tags::FaceCoordinatesCompute<3, Frame::Grid, false>,
            Tags::FaceCoordinatesCompute<3, Frame::Inertial, true>,
            Tags::FaceCoordinatesCompute<3, Frame::Inertial, false>>>(
        initial_time, excision_sphere, domain_creator->functions_of_time(),
        element, mesh, grid_coords,
        my_block.moving_mesh_grid_to_inertial_map().get_clone());
    const auto centered_grid_1 =
        db::get<Tags::FaceCoordinatesCompute<3, Frame::Grid, true>>(box);
    const auto uncentered_grid_1 =
        db::get<Tags::FaceCoordinatesCompute<3, Frame::Grid, false>>(box);
    const auto centered_inertial_1 =
        db::get<Tags::FaceCoordinatesCompute<3, Frame::Inertial, true>>(box);
    const auto uncentered_inertial_1 =
        db::get<Tags::FaceCoordinatesCompute<3, Frame::Inertial, false>>(box);
    if (all_faces_grid_coords.count(element_id)) {
      CHECK(centered_grid_1.has_value());
      CHECK(uncentered_grid_1.has_value());
      CHECK(centered_inertial_1.has_value());
      CHECK(uncentered_inertial_1.has_value());

      CHECK_ITERABLE_APPROX(centered_grid_1.value(),
                            all_faces_grid_coords.at(element_id));
      for (size_t i = 0; i < 3; ++i) {
        CHECK_ITERABLE_APPROX(centered_grid_1->get(i),
                              centered_inertial_1->get(i));
        CHECK_ITERABLE_APPROX(uncentered_grid_1->get(i),
                              uncentered_inertial_1->get(i));
        CHECK_ITERABLE_APPROX(
            centered_grid_1->get(i),
            uncentered_grid_1->get(i) - excision_sphere.center().get(i));
      }
    } else {
      CHECK(not centered_grid_1.has_value());
      CHECK(not uncentered_grid_1.has_value());
      CHECK(not centered_inertial_1.has_value());
      CHECK(not uncentered_inertial_1.has_value());
    }

    const double new_time = 5.;
    ::Initialization::mutate_assign<tmpl::list<::Tags::Time>>(
        make_not_null(&box), new_time);

    const auto centered_grid_2 =
        db::get<Tags::FaceCoordinatesCompute<3, Frame::Grid, true>>(box);
    const auto uncentered_grid_2 =
        db::get<Tags::FaceCoordinatesCompute<3, Frame::Grid, false>>(box);
    const auto centered_inertial_2 =
        db::get<Tags::FaceCoordinatesCompute<3, Frame::Inertial, true>>(box);
    const auto uncentered_inertial_2 =
        db::get<Tags::FaceCoordinatesCompute<3, Frame::Inertial, false>>(box);
    if (all_faces_grid_coords.count(element_id)) {
      CHECK(centered_grid_2.has_value());
      CHECK(uncentered_grid_2.has_value());
      CHECK(centered_inertial_2.has_value());
      CHECK(uncentered_grid_2.has_value());

      CHECK_ITERABLE_APPROX(centered_grid_2.value(), centered_grid_1.value());
      CHECK_ITERABLE_APPROX(uncentered_grid_2.value(),
                            uncentered_grid_1.value());
      const auto& inertial_particle_position =
          db::get<Tags::InertialParticlePosition<Dim>>(box);
      const auto& grid_to_inertial_map =
          db::get<::domain::CoordinateMaps::Tags::CoordinateMap<
              Dim, Frame::Grid, Frame::Inertial>>(box);
      const auto& functions_of_time =
          db::get<domain::Tags::FunctionsOfTimeInitialize>(box);
      CHECK_ITERABLE_APPROX(uncentered_inertial_2.value(),
                            grid_to_inertial_map(uncentered_grid_2.value(),
                                                 new_time, functions_of_time));
      for (size_t i = 0; i < Dim; ++i) {
        CHECK_ITERABLE_APPROX(
            centered_inertial_2->get(i),
            uncentered_inertial_2->get(i) - inertial_particle_position.get(i));
      }
    } else {
      CHECK(not centered_grid_2.has_value());
      CHECK(not uncentered_grid_2.has_value());
      CHECK(not centered_inertial_2.has_value());
      CHECK(not uncentered_inertial_2.has_value());
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

void test_puncture_field() {
  static constexpr size_t Dim = 3;
  ::TestHelpers::db::test_compute_tag<Tags::PunctureFieldCompute<Dim>>(
      "PunctureField");
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1., 1.);
  tnsr::I<double, Dim, Frame::Grid> charge_pos{{7.1, 0., 0.}};
  const ::ExcisionSphere excision_sphere{1., charge_pos, {}};
  const double orbital_radius = get(magnitude(charge_pos));
  auto random_points =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          make_not_null(&gen), dist, DataVector(50));
  random_points.get(0) += charge_pos.get(0);

  const double time = 1.;
  const size_t order = 0;
  const auto box_abutting = db::create<
      db::AddSimpleTags<Tags::FaceCoordinates<Dim, Frame::Inertial, false>,
                        Tags::ExcisionSphere<Dim>, ::Tags::Time,
                        Tags::ExpansionOrder>,
      db::AddComputeTags<Tags::PunctureFieldCompute<Dim>>>(
      std::make_optional<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          random_points),
      excision_sphere, time, order);
  const auto singular_field = get<Tags::PunctureField<Dim>>(box_abutting);

  CHECK(singular_field.has_value());
  Variables<tmpl::list<CurvedScalarWave::Tags::Psi,
                       ::Tags::dt<CurvedScalarWave::Tags::Psi>,
                       ::Tags::deriv<CurvedScalarWave::Tags::Psi,
                                     tmpl::size_t<3>, Frame::Inertial>>>
      expected{};
  puncture_field(make_not_null(&expected), random_points, time, orbital_radius,
                 time, order);
  CHECK_VARIABLES_APPROX(singular_field.value(), expected);
  std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> nullopt{};
  const auto box_not_abutting = db::create<
      db::AddSimpleTags<Tags::FaceCoordinates<Dim, Frame::Inertial, false>,
                        Tags::ExcisionSphere<Dim>, ::Tags::Time,
                        Tags::ExpansionOrder>,
      db::AddComputeTags<Tags::PunctureFieldCompute<Dim>>>(
      nullopt, excision_sphere, time, order);
  const auto puncture_field_nullopt =
      get<Tags::PunctureField<Dim>>(box_not_abutting);
  CHECK(not puncture_field_nullopt.has_value());
}

void test_check_input_file() {
  const auto bbh_correct =
      TestHelpers::CurvedScalarWave::Worldtube::worldtube_binary_compact_object(
          7., 0.2, pow(7., -1.5));
  const gr::Solutions::KerrSchild kerr_schild_correct(
      1., make_array(0., 0., 0.), make_array(0., 0., 0.));
  CHECK(Tags::CheckInputFile<3, gr::Solutions::KerrSchild>::create_from_options(
      bbh_correct, "ExcisionSphereA", kerr_schild_correct));
  {
    const auto bbh_incorrect = TestHelpers::CurvedScalarWave::Worldtube::
        worldtube_binary_compact_object(7., 0.2, 1.);
    CHECK_THROWS_WITH(
        (Tags::CheckInputFile<3, gr::Solutions::KerrSchild>::
             create_from_options(bbh_incorrect, "ExcisionSphereA",
                                 kerr_schild_correct)),
        Catch::Matchers::Contains(
            "Only circular orbits are implemented at the moment so the "
            "angular velocity should be [0., 0., orbital_radius^(-3/2)] = "));
  }
  {
    const std::unique_ptr<DomainCreator<3>> shell =
        std::make_unique<domain::creators::Sphere>(
            1.5, 3., domain::creators::Sphere::Excision{}, size_t(1), size_t(6),
            true);
    CHECK_THROWS_WITH(
        (Tags::CheckInputFile<3, gr::Solutions::KerrSchild>::
             create_from_options(shell, "ExcisionSphere", kerr_schild_correct)),
        Catch::Matchers::Contains(
            "Expected functions of time to contain 'Rotation'."));
  }
  {
    const gr::Solutions::KerrSchild kerr_schild_off_center(
        1., make_array(0., 0., 0.), make_array(0.1, 0., 0.));
    CHECK_THROWS_WITH(
        (Tags::CheckInputFile<3, gr::Solutions::KerrSchild>::
             create_from_options(bbh_correct, "ExcisionSphere",
                                 kerr_schild_off_center)),
        Catch::Matchers::Contains(
            "The central black hole must be centered at [0., 0., 0.]."));
  }
  {
    const gr::Solutions::KerrSchild kerr_schild_spinning(
        1., make_array(0.1, 0., 0.), make_array(0., 0., 0.));
    CHECK_THROWS_WITH(
        (Tags::CheckInputFile<3, gr::Solutions::KerrSchild>::
             create_from_options(bbh_correct, "ExcisionSphere",
                                 kerr_schild_spinning)),
        Catch::Matchers::Contains("Black hole spin is not implemented yet but "
                                  "you requested non-zero spin."));
  }
  {
    const gr::Solutions::KerrSchild kerr_schild_2M(2., make_array(0., 0., 0.),
                                                   make_array(0., 0., 0.));
    CHECK_THROWS_WITH(
        (Tags::CheckInputFile<
            3, gr::Solutions::KerrSchild>::create_from_options(bbh_correct,
                                                               "ExcisionSphere",
                                                               kerr_schild_2M)),
        Catch::Matchers::Contains("The central black hole must have mass 1."));
  }
}

}  // namespace

// [[TimeOut, 15]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.Worldtube.Tags",
                  "[Unit][Evolution]") {
  CHECK(TestHelpers::test_option_tag<OptionTags::ExcisionSphere>("Worldtube") ==
        "Worldtube");
  CHECK(TestHelpers::test_option_tag<
            CurvedScalarWave::Worldtube::OptionTags::ExpansionOrder>("0") == 0);
  TestHelpers::db::test_simple_tag<Tags::ExcisionSphere<3>>("ExcisionSphere");
  TestHelpers::db::test_simple_tag<
      CurvedScalarWave::Worldtube::Tags::ExpansionOrder>("ExpansionOrder");
  TestHelpers::db::test_simple_tag<Tags::ElementFacesGridCoordinates<3>>(
      "ElementFacesGridCoordinates");
  TestHelpers::db::test_simple_tag<Tags::FaceCoordinates<3, Frame::Grid, true>>(
      "FaceCoordinates");
  TestHelpers::db::test_simple_tag<
      Tags::FaceCoordinates<3, Frame::Grid, false>>("FaceCoordinates");
  TestHelpers::db::test_simple_tag<
      Tags::FaceCoordinates<3, Frame::Inertial, true>>("FaceCoordinates");
  TestHelpers::db::test_simple_tag<
      Tags::FaceCoordinates<3, Frame::Inertial, false>>("FaceCoordinates");
  TestHelpers::db::test_simple_tag<Tags::InertialParticlePosition<3>>(
      "InertialParticlePosition");
  TestHelpers::db::test_simple_tag<Tags::PunctureField<3>>("PunctureField");
  TestHelpers::db::test_simple_tag<
      Tags::CheckInputFile<3, gr::Solutions::KerrSchild>>("CheckInputFile");
  TestHelpers::db::test_simple_tag<Tags::RegularFieldAdvectiveTerm<3>>(
      "RegularFieldAdvectiveTerm");
  TestHelpers::db::test_simple_tag<Tags::ObserveCoefficientsTrigger>(
      "ObserveCoefficientsTrigger");
  test_excision_sphere_tag();
  test_compute_face_coordinates_grid();
  test_compute_face_coordinates();
  test_inertial_particle_position_compute();
  test_puncture_field();
  test_check_input_file();
}
}  // namespace CurvedScalarWave::Worldtube
