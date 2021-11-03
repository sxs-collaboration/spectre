// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare domain::CoordinateMaps::Rotation
// IWYU pragma: no_forward_declare Tensor

namespace domain {
namespace {
template <size_t Dim>
struct Directions : db::SimpleTag {
  static constexpr size_t volume_dim = Dim;
  static std::string name() { return "Directions"; }
  using type = std::unordered_set<Direction<Dim>>;
};

template <typename Map>
void check(const Map& map,
           const std::array<std::array<double, Map::dim>, Map::dim>& expected) {
  const Mesh<Map::dim - 1> mesh{3, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
  const auto num_grid_points = mesh.number_of_grid_points();
  for (size_t d = 0; d < Map::dim; ++d) {
    const auto upper_normal = unnormalized_face_normal(
        mesh, map, Direction<Map::dim>(d, Side::Upper));
    const auto lower_normal = unnormalized_face_normal(
        mesh, map, Direction<Map::dim>(d, Side::Lower));
    for (size_t i = 0; i < 2; ++i) {
      CHECK_ITERABLE_APPROX(
          upper_normal.get(i),
          DataVector(num_grid_points, gsl::at(gsl::at(expected, d), i)));
      CHECK_ITERABLE_APPROX(
          lower_normal.get(i),
          DataVector(num_grid_points, -gsl::at(gsl::at(expected, d), i)));
    }
  }
}

template <size_t Dim>
std::unordered_set<Direction<Dim>> get_directions();

template <>
std::unordered_set<Direction<1>> get_directions<1>() {
  return std::unordered_set<Direction<1>>{Direction<1>::upper_xi()};
}

template <>
std::unordered_set<Direction<2>> get_directions<2>() {
  return std::unordered_set<Direction<2>>{Direction<2>::upper_xi(),
                                          Direction<2>::lower_eta()};
}

template <>
std::unordered_set<Direction<3>> get_directions<3>() {
  return std::unordered_set<Direction<3>>{Direction<3>::upper_xi(),
                                          Direction<3>::lower_eta(),
                                          Direction<3>::lower_zeta()};
}

template <size_t Dim>
void check_time_dependent(
    const ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
    const std::unique_ptr<CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>>&
        grid_to_inertial_map,
    const std::unique_ptr<
        CoordinateMapBase<Frame::ElementLogical, Frame::Inertial, Dim>>&
        logical_to_inertial_map,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  const Mesh<Dim - 1> interface_mesh{3, Spectral::Basis::Legendre,
                                     Spectral::Quadrature::GaussLobatto};
  for (size_t d = 0; d < Dim; ++d) {
    const auto upper_normal = unnormalized_face_normal(
        interface_mesh, logical_to_grid_map, *grid_to_inertial_map, time,
        functions_of_time, Direction<Dim>(d, Side::Upper));
    const auto lower_normal = unnormalized_face_normal(
        interface_mesh, logical_to_grid_map, *grid_to_inertial_map, time,
        functions_of_time, Direction<Dim>(d, Side::Lower));

    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>
        inv_jacobian_upper = logical_to_inertial_map->inv_jacobian(
            interface_logical_coordinates(interface_mesh,
                                          Direction<Dim>(d, Side::Upper)),
            time, functions_of_time);
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>
        inv_jacobian_lower = logical_to_inertial_map->inv_jacobian(
            interface_logical_coordinates(interface_mesh,
                                          Direction<Dim>(d, Side::Lower)),
            time, functions_of_time);

    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(upper_normal.get(i),
                            DataVector{inv_jacobian_upper.get(d, i)});
      CHECK_ITERABLE_APPROX(lower_normal.get(i),
                            DataVector{-inv_jacobian_lower.get(d, i)});
    }
  }

  // Now check the compute items
  const auto box = db::create<
      db::AddSimpleTags<Tags::Element<Dim>, Directions<Dim>, Tags::Mesh<Dim>,
                        Tags::ElementMap<Dim, Frame::Grid>,
                        CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                            Frame::Inertial>,
                        ::Tags::Time, Tags::FunctionsOfTimeInitialize>,
      db::AddComputeTags<
          Tags::BoundaryDirectionsExteriorCompute<Dim>,
          Tags::InterfaceCompute<Directions<Dim>, Tags::Direction<Dim>>,
          Tags::InterfaceCompute<Directions<Dim>, Tags::InterfaceMesh<Dim>>,
          Tags::InterfaceCompute<Tags::BoundaryDirectionsExterior<Dim>,
                                 Tags::Direction<Dim>>,
          Tags::InterfaceCompute<Tags::BoundaryDirectionsExterior<Dim>,
                                 Tags::InterfaceMesh<Dim>>,
          Tags::InterfaceCompute<
              Tags::BoundaryDirectionsExterior<Dim>,
              Tags::UnnormalizedFaceNormalMovingMeshCompute<Dim>>,
          Tags::InterfaceCompute<
              Directions<Dim>,
              Tags::UnnormalizedFaceNormalMovingMeshCompute<Dim>>>>(
      Element<Dim>(logical_to_grid_map.element_id(), {}), get_directions<Dim>(),
      Mesh<Dim>{3, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto},
      ElementMap<Dim, Frame::Grid>(logical_to_grid_map.element_id(),
                                   logical_to_grid_map.block_map().get_clone()),
      grid_to_inertial_map->get_clone(), time,
      clone_unique_ptrs(functions_of_time));

  TestHelpers::db::test_compute_tag<
      Tags::InterfaceCompute<Tags::BoundaryDirectionsExterior<2>,
                             Tags::UnnormalizedFaceNormalMovingMeshCompute<2>>>(
      "BoundaryDirectionsExterior<UnnormalizedFaceNormal>"s);

  for (const auto& direction_and_face_normal :
       db::get<Tags::Interface<Tags::BoundaryDirectionsExterior<Dim>,
                               Tags::UnnormalizedFaceNormal<Dim>>>(box)) {
    auto expected_normal = unnormalized_face_normal(
        interface_mesh, logical_to_grid_map, *grid_to_inertial_map, time,
        functions_of_time, direction_and_face_normal.first);
    for (size_t i = 0; i < expected_normal.size(); ++i) {
      expected_normal.get(i) *= -1.0;
    }

    CHECK_ITERABLE_APPROX(expected_normal, direction_and_face_normal.second);
  }

  for (const auto& direction_and_face_normal : db::get<
           Tags::Interface<Directions<Dim>, Tags::UnnormalizedFaceNormal<Dim>>>(
           box)) {
    const auto expected_normal = unnormalized_face_normal(
        interface_mesh, logical_to_grid_map, *grid_to_inertial_map, time,
        functions_of_time, direction_and_face_normal.first);

    CHECK_ITERABLE_APPROX(expected_normal, direction_and_face_normal.second);
  }
}

void test_face_normal_coordinate_map() {
  INFO("Test coordinate map");
  // [face_normal_example]
  const Mesh<0> mesh_0d;
  const auto map_1d = make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
      CoordinateMaps::Affine(-1.0, 1.0, -3.0, 7.0));
  const auto normal_1d_lower =
      unnormalized_face_normal(mesh_0d, map_1d, Direction<1>::lower_xi());
  // [face_normal_example]

  CHECK(normal_1d_lower.get(0) == DataVector(1, -0.2));

  const auto normal_1d_upper =
      unnormalized_face_normal(mesh_0d, map_1d, Direction<1>::upper_xi());

  CHECK(normal_1d_upper.get(0) == DataVector(1, 0.2));

  check(make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
            CoordinateMaps::Rotation<2>(atan2(4., 3.))),
        {{{{0.6, 0.8}}, {{-0.8, 0.6}}}});

  check(make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                           CoordinateMaps::Rotation<2>>(
                {-1., 1., 2., 7.}, CoordinateMaps::Rotation<2>(atan2(4., 3.)))),
        {{{{0.4, 0., 0.}}, {{0., 0.6, 0.8}}, {{0., -0.8, 0.6}}}});
}

template <typename TargetFrame>
void test_face_normal_element_map() {
  const Mesh<0> mesh_0d;
  const auto map_1d = ElementMap<1, TargetFrame>(
      ElementId<1>{0},
      make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
          CoordinateMaps::Affine(-1.0, 1.0, -3.0, 7.0)));
  const auto normal_1d_lower =
      unnormalized_face_normal(mesh_0d, map_1d, Direction<1>::lower_xi());

  CHECK(normal_1d_lower.get(0) == DataVector(1, -0.2));

  const auto normal_1d_upper =
      unnormalized_face_normal(mesh_0d, map_1d, Direction<1>::upper_xi());

  CHECK(normal_1d_upper.get(0) == DataVector(1, 0.2));

  check(ElementMap<2, TargetFrame>(
            ElementId<2>(0),
            make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
                CoordinateMaps::Rotation<2>(atan2(4., 3.)))),
        {{{{0.6, 0.8}}, {{-0.8, 0.6}}}});

  check(ElementMap<3, TargetFrame>(
            ElementId<3>(0),
            make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
                CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                               CoordinateMaps::Rotation<2>>(
                    {-1., 1., 2., 7.},
                    CoordinateMaps::Rotation<2>(atan2(4., 3.))))),
        {{{{0.4, 0., 0.}}, {{0., 0.6, 0.8}}, {{0., -0.8, 0.6}}}});
}

void test_face_normal_moving_mesh() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);

  const std::array<double, 4> times_to_check{{0.0, 0.3, 1.1, 7.8}};
  const double outer_boundary = 10.0;
  std::array<std::string, 2> functions_of_time_names{
      {"ExpansionA", "ExpansionB"}};
  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<2>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
  std::unordered_map<std::string, FoftPtr> functions_of_time{};
  const std::array<DataVector, 3> init_func_a{{{1.0}, {-0.1}, {0.0}}};
  const std::array<DataVector, 3> init_func_b{{{1.0}, {0.0}, {0.0}}};
  const double initial_time = 0.0;
  const double expiration_time = 10.0;
  functions_of_time["ExpansionA"] =
      std::make_unique<Polynomial>(initial_time, init_func_a, expiration_time);
  functions_of_time["ExpansionB"] =
      std::make_unique<Polynomial>(initial_time, init_func_b, expiration_time);

  {
    INFO("1d");
    const ElementMap<1, Frame::Grid> logical_to_grid_map(
        ElementId<1>(0),
        make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
            CoordinateMaps::Affine(-1.0, 1.0, 2.0, 7.8)));
    const auto grid_to_inertial_map =
        make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::CubicScale<1>{
                outer_boundary, functions_of_time_names[0],
                functions_of_time_names[1]});
    const auto logical_to_inertial_map =
        make_coordinate_map_base<Frame::ElementLogical, Frame::Inertial>(
            CoordinateMaps::Affine(-1.0, 1.0, 2.0, 7.8),
            CoordinateMaps::TimeDependent::CubicScale<1>{
                outer_boundary, functions_of_time_names[0],
                functions_of_time_names[1]});

    for (const double time : times_to_check) {
      check_time_dependent(logical_to_grid_map, grid_to_inertial_map,
                           logical_to_inertial_map, time, functions_of_time);
    }
  }
  {
    INFO("2d");
    const ElementMap<2, Frame::Grid> logical_to_grid_map(
        ElementId<2>(0),
        make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
            CoordinateMaps::Rotation<2>(atan2(4., 3.))));
    const auto grid_to_inertial_map =
        make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::CubicScale<2>{
                outer_boundary, functions_of_time_names[0],
                functions_of_time_names[1]});
    const auto logical_to_inertial_map =
        make_coordinate_map_base<Frame::ElementLogical, Frame::Inertial>(
            CoordinateMaps::Rotation<2>(atan2(4., 3.)),
            CoordinateMaps::TimeDependent::CubicScale<2>{
                outer_boundary, functions_of_time_names[0],
                functions_of_time_names[1]});

    for (const double time : times_to_check) {
      check_time_dependent(logical_to_grid_map, grid_to_inertial_map,
                           logical_to_inertial_map, time, functions_of_time);
    }
  }
  {
    INFO("3d");
    const ElementMap<3, Frame::Grid> logical_to_grid_map(
        ElementId<3>(0),
        make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                           CoordinateMaps::Rotation<2>>(
                {-1., 1., 2., 7.},
                CoordinateMaps::Rotation<2>(atan2(4., 3.)))));
    const auto grid_to_inertial_map =
        make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            CoordinateMaps::TimeDependent::CubicScale<3>{
                outer_boundary, functions_of_time_names[0],
                functions_of_time_names[1]});
    const auto logical_to_inertial_map =
        make_coordinate_map_base<Frame::ElementLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                           CoordinateMaps::Rotation<2>>(
                {-1., 1., 2., 7.}, CoordinateMaps::Rotation<2>(atan2(4., 3.))),
            CoordinateMaps::TimeDependent::CubicScale<3>{
                outer_boundary, functions_of_time_names[0],
                functions_of_time_names[1]});

    for (const double time : times_to_check) {
      check_time_dependent(logical_to_grid_map, grid_to_inertial_map,
                           logical_to_inertial_map, time, functions_of_time);
    }
  }
}

void test_compute_item() {
  INFO("Testing compute item");
  const auto box = db::create<
      db::AddSimpleTags<Tags::Element<2>, Directions<2>, Tags::Mesh<2>,
                        Tags::ElementMap<2>>,
      db::AddComputeTags<
          Tags::BoundaryDirectionsExteriorCompute<2>,
          Tags::InterfaceCompute<Directions<2>, Tags::Direction<2>>,
          Tags::InterfaceCompute<Directions<2>, Tags::InterfaceMesh<2>>,
          Tags::InterfaceCompute<Tags::BoundaryDirectionsExterior<2>,
                                 Tags::Direction<2>>,
          Tags::InterfaceCompute<Tags::BoundaryDirectionsExterior<2>,
                                 Tags::InterfaceMesh<2>>,
          Tags::InterfaceCompute<Tags::BoundaryDirectionsExterior<2>,
                                 Tags::UnnormalizedFaceNormalCompute<2>>,
          Tags::InterfaceCompute<Directions<2>,
                                 Tags::UnnormalizedFaceNormalCompute<2>>>>(
      Element<2>(ElementId<2>(0), {}),
      std::unordered_set<Direction<2>>{Direction<2>::upper_xi(),
                                       Direction<2>::lower_eta()},
      Mesh<2>{2, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto},
      ElementMap<2, Frame::Inertial>(
          ElementId<2>(0),
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              CoordinateMaps::Rotation<2>(atan2(4., 3.)))));

  TestHelpers::db::test_compute_tag<
      Tags::InterfaceCompute<Tags::BoundaryDirectionsExterior<2>,
                             Tags::UnnormalizedFaceNormalCompute<2>>>(
      "BoundaryDirectionsExterior<UnnormalizedFaceNormal>"s);

  std::unordered_map<Direction<2>, tnsr::i<DataVector, 2>> expected;
  expected[Direction<2>::upper_xi()] =
      tnsr::i<DataVector, 2>{{{{0.6, 0.6}, {0.8, 0.8}}}};
  expected[Direction<2>::lower_eta()] =
      tnsr::i<DataVector, 2>{{{{0.8, 0.8}, {-0.6, -0.6}}}};

  std::unordered_map<Direction<2>, tnsr::i<DataVector, 2>>
      expected_external_normal;
  expected_external_normal[Direction<2>::lower_xi()] =
      tnsr::i<DataVector, 2>{{{{0.6, 0.6}, {0.8, 0.8}}}};
  expected_external_normal[Direction<2>::upper_xi()] =
      tnsr::i<DataVector, 2>{{{{-0.6, -0.6}, {-0.8, -0.8}}}};
  expected_external_normal[Direction<2>::lower_eta()] =
      tnsr::i<DataVector, 2>{{{{-0.8, -0.8}, {0.6, 0.6}}}};
  expected_external_normal[Direction<2>::upper_eta()] =
      tnsr::i<DataVector, 2>{{{{0.8, 0.8}, {-0.6, -0.6}}}};

  CHECK_ITERABLE_APPROX(
      (get<Tags::Interface<Directions<2>, Tags::UnnormalizedFaceNormal<2>>>(
          box)),
      expected);
  CHECK_ITERABLE_APPROX(
      (get<Tags::Interface<Tags::BoundaryDirectionsExterior<2>,
                           Tags::UnnormalizedFaceNormal<2>>>(box)),
      expected_external_normal);

  // Now test the external normals with a non-affine map, to ensure that the
  // exterior face normal is the inverted interior one
  const auto box_with_non_affine_map = db::create<
      db::AddSimpleTags<Tags::Element<2>, Tags::Mesh<2>, Tags::ElementMap<2>>,
      db::AddComputeTags<
          Tags::BoundaryDirectionsExteriorCompute<2>,
          Tags::BoundaryDirectionsInteriorCompute<2>,
          Tags::InterfaceCompute<Tags::BoundaryDirectionsExterior<2>,
                                 Tags::Direction<2>>,
          Tags::InterfaceCompute<Tags::BoundaryDirectionsExterior<2>,
                                 Tags::InterfaceMesh<2>>,
          Tags::InterfaceCompute<Tags::BoundaryDirectionsExterior<2>,
                                 Tags::UnnormalizedFaceNormalCompute<2>>,
          Tags::InterfaceCompute<Tags::BoundaryDirectionsInterior<2>,
                                 Tags::Direction<2>>,
          Tags::InterfaceCompute<Tags::BoundaryDirectionsInterior<2>,
                                 Tags::InterfaceMesh<2>>,
          Tags::InterfaceCompute<Tags::BoundaryDirectionsInterior<2>,
                                 Tags::UnnormalizedFaceNormalCompute<2>>>>(
      Element<2>(ElementId<2>(0), {}),
      Mesh<2>{2, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto},
      ElementMap<2, Frame::Inertial>(
          ElementId<2>(0),
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              CoordinateMaps::Wedge<2>(1., 2., 0., 1., OrientationMap<2>{},
                                       false))));

  auto invert = [](std::unordered_map<Direction<2>, tnsr::i<DataVector, 2>>
                       map_of_vectors) {
    for (auto& vector : map_of_vectors) {
      for (auto& dv : vector.second) {
        dv *= -1.;
      }
    }
    return map_of_vectors;
  };

  CHECK((db::get<Tags::Interface<Tags::BoundaryDirectionsExterior<2>,
                                 Tags::UnnormalizedFaceNormal<2>>>(
            box_with_non_affine_map)) ==
        (invert(db::get<Tags::Interface<Tags::BoundaryDirectionsInterior<2>,
                                        Tags::UnnormalizedFaceNormal<2>>>(
            box_with_non_affine_map))));
}

template <size_t Dim, typename Frame>
void test_tags() {
  TestHelpers::db::test_simple_tag<Tags::UnnormalizedFaceNormal<Dim, Frame>>(
      "UnnormalizedFaceNormal");
  TestHelpers::db::test_compute_tag<
      Tags::UnnormalizedFaceNormalCompute<Dim, Frame>>(
      "UnnormalizedFaceNormal");
  if constexpr (std::is_same_v<Frame, ::Frame::Inertial>) {
    TestHelpers::db::test_compute_tag<
        Tags::UnnormalizedFaceNormalMovingMeshCompute<Dim>>(
        "UnnormalizedFaceNormal");
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FaceNormal", "[Unit][Domain]") {
  test_face_normal_coordinate_map();
  test_face_normal_element_map<Frame::Inertial>();
  test_face_normal_element_map<Frame::Grid>();
  test_face_normal_moving_mesh();
  test_compute_item();
  test_tags<1, Frame::Inertial>();
  test_tags<2, Frame::Inertial>();
  test_tags<3, Frame::Inertial>();
  test_tags<1, Frame::Grid>();
  test_tags<2, Frame::Grid>();
  test_tags<3, Frame::Grid>();
}
}  // namespace domain
