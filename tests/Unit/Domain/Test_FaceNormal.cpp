// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Side.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare domain::CoordinateMaps::Rotation
// IWYU pragma: no_forward_declare Tensor

namespace {
template <typename Map>
void check(const Map& map,
           const std::array<std::array<double, Map::dim>, Map::dim>& expected) {
  const domain::Mesh<Map::dim - 1> mesh{3, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto};
  const auto num_grid_points = mesh.number_of_grid_points();
  for (size_t d = 0; d < Map::dim; ++d) {
    const auto upper_normal = unnormalized_face_normal(
        mesh, map, domain::Direction<Map::dim>(d, domain::Side::Upper));
    const auto lower_normal = unnormalized_face_normal(
        mesh, map, domain::Direction<Map::dim>(d, domain::Side::Lower));
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
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FaceNormal.CoordMap", "[Unit][Domain]") {
  /// [face_normal_example]
  const domain::Mesh<0> mesh_0d;
  const auto map_1d = domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
      domain::CoordinateMaps::Affine(-1.0, 1.0, -3.0, 7.0));
  const auto normal_1d_lower = unnormalized_face_normal(
      mesh_0d, map_1d, domain::Direction<1>::lower_xi());
  /// [face_normal_example]

  CHECK(normal_1d_lower.get(0) == DataVector(1, -0.2));

  const auto normal_1d_upper = unnormalized_face_normal(
      mesh_0d, map_1d, domain::Direction<1>::upper_xi());

  CHECK(normal_1d_upper.get(0) == DataVector(1, 0.2));

  check(domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
            domain::CoordinateMaps::Rotation<2>(atan2(4., 3.))),
        {{{{0.6, 0.8}}, {{-0.8, 0.6}}}});

  check(domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
            domain::CoordinateMaps::ProductOf2Maps<
                domain::CoordinateMaps::Affine,
                domain::CoordinateMaps::Rotation<2>>(
                {-1., 1., 2., 7.},
                domain::CoordinateMaps::Rotation<2>(atan2(4., 3.)))),
        {{{{0.4, 0., 0.}}, {{0., 0.6, 0.8}}, {{0., -0.8, 0.6}}}});
}

namespace {
template <typename TargetFrame>
void test_face_normal_element_map() {
  const domain::Mesh<0> mesh_0d;
  const auto map_1d = domain::ElementMap<1, TargetFrame>(
      domain::ElementId<1>{0},
      domain::make_coordinate_map_base<Frame::Logical, TargetFrame>(
          domain::CoordinateMaps::Affine(-1.0, 1.0, -3.0, 7.0)));
  const auto normal_1d_lower = unnormalized_face_normal(
      mesh_0d, map_1d, domain::Direction<1>::lower_xi());

  CHECK(normal_1d_lower.get(0) == DataVector(1, -0.2));

  const auto normal_1d_upper = unnormalized_face_normal(
      mesh_0d, map_1d, domain::Direction<1>::upper_xi());

  CHECK(normal_1d_upper.get(0) == DataVector(1, 0.2));

  check(domain::ElementMap<2, TargetFrame>(
            domain::ElementId<2>(0),
            domain::make_coordinate_map_base<Frame::Logical, TargetFrame>(
                domain::CoordinateMaps::Rotation<2>(atan2(4., 3.)))),
        {{{{0.6, 0.8}}, {{-0.8, 0.6}}}});

  check(domain::ElementMap<3, TargetFrame>(
            domain::ElementId<3>(0),
            domain::make_coordinate_map_base<Frame::Logical, TargetFrame>(
                domain::CoordinateMaps::ProductOf2Maps<
                    domain::CoordinateMaps::Affine,
                    domain::CoordinateMaps::Rotation<2>>(
                    {-1., 1., 2., 7.},
                    domain::CoordinateMaps::Rotation<2>(atan2(4., 3.))))),
        {{{{0.4, 0., 0.}}, {{0., 0.6, 0.8}}, {{0., -0.8, 0.6}}}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FaceNormal.ElementMap", "[Unit][Domain]") {
  test_face_normal_element_map<Frame::Inertial>();
  test_face_normal_element_map<Frame::Grid>();
}

namespace {
struct Directions : db::SimpleTag {
  static std::string name() noexcept { return "Directions"; }
  using type = std::unordered_set<domain::Direction<2>>;
};
}  // namespace
SPECTRE_TEST_CASE("Unit.Domain.FaceNormal.ComputeItem", "[Unit][Domain]") {
  const auto box = db::create<
      db::AddSimpleTags<Directions, domain::Tags::Mesh<2>,
                        domain::Tags::ElementMap<2>>,
      db::AddComputeTags<
          domain::Tags::Interface<Directions, domain::Tags::Direction<2>>,
          domain::Tags::Interface<Directions, domain::Tags::Mesh<1>>,
          domain::Tags::Interface<Directions,
                                  domain::Tags::UnnormalizedFaceNormal<2>>>>(
      std::unordered_set<domain::Direction<2>>{
          domain::Direction<2>::upper_xi(), domain::Direction<2>::lower_eta()},
      domain::Mesh<2>{2, Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto},
      domain::ElementMap<2, Frame::Inertial>(
          domain::ElementId<2>(0),
          domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              domain::CoordinateMaps::Rotation<2>(atan2(4., 3.)))));

  std::unordered_map<domain::Direction<2>, tnsr::i<DataVector, 2>> expected;
  expected[domain::Direction<2>::upper_xi()] =
      tnsr::i<DataVector, 2>{{{{0.6, 0.6}, {0.8, 0.8}}}};
  expected[domain::Direction<2>::lower_eta()] =
      tnsr::i<DataVector, 2>{{{{0.8, 0.8}, {-0.6, -0.6}}}};

  CHECK_ITERABLE_APPROX(
      (get<domain::Tags::Interface<
           Directions, domain::Tags::UnnormalizedFaceNormal<2>>>(box)),
      expected);
}
