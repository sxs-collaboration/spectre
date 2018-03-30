// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Side.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare CoordinateMaps::Rotation
// IWYU pragma: no_forward_declare Tensor

namespace {
  template <typename Map>
  void check(
      const Map& map,
      const std::array<std::array<double, Map::dim>, Map::dim>& expected) {
    const Index<Map::dim - 1> extents(3);
    for (size_t d = 0; d < Map::dim; ++d) {
      const auto upper_normal = unnormalized_face_normal(
          extents, map, Direction<Map::dim>(d, Side::Upper));
      const auto lower_normal = unnormalized_face_normal(
          extents, map, Direction<Map::dim>(d, Side::Lower));
      for (size_t i = 0; i < 2; ++i) {
        CHECK_ITERABLE_APPROX(
            upper_normal.get(i),
            DataVector(extents.product(), gsl::at(gsl::at(expected, d), i)));
        CHECK_ITERABLE_APPROX(
            lower_normal.get(i),
            DataVector(extents.product(), -gsl::at(gsl::at(expected, d), i)));
      }
    }
  }
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FaceNormal.CoordMap", "[Unit][Domain]") {
  /// [face_normal_example]
  const Index<0> extents_0d;
  const auto map_1d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      CoordinateMaps::Affine(-1.0, 1.0, -3.0, 7.0));
  const auto normal_1d_lower =
      unnormalized_face_normal(extents_0d, map_1d, Direction<1>::lower_xi());
  /// [face_normal_example]

  CHECK(normal_1d_lower.get(0) == DataVector(1, -0.2));

  const auto normal_1d_upper =
      unnormalized_face_normal(extents_0d, map_1d, Direction<1>::upper_xi());

  CHECK(normal_1d_upper.get(0) == DataVector(1, 0.2));

  check(make_coordinate_map<Frame::Logical, Frame::Grid>(
            CoordinateMaps::Rotation<2>(atan2(4., 3.))),
        {{{{0.6, 0.8}}, {{-0.8, 0.6}}}});

  check(make_coordinate_map<Frame::Logical, Frame::Grid>(
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                           CoordinateMaps::Rotation<2>>(
                {-1., 1., 2., 7.}, CoordinateMaps::Rotation<2>(atan2(4., 3.)))),
        {{{{0.4, 0., 0.}}, {{0., 0.6, 0.8}}, {{0., -0.8, 0.6}}}});
}

namespace {
template <typename TargetFrame>
void test_face_normal_element_map() {
  const Index<0> extents_0d;
  const auto map_1d = ElementMap<1, TargetFrame>(
      ElementId<1>{0}, make_coordinate_map_base<Frame::Logical, TargetFrame>(
                           CoordinateMaps::Affine(-1.0, 1.0, -3.0, 7.0)));
  const auto normal_1d_lower =
      unnormalized_face_normal(extents_0d, map_1d, Direction<1>::lower_xi());

  CHECK(normal_1d_lower.get(0) == DataVector(1, -0.2));

  const auto normal_1d_upper =
      unnormalized_face_normal(extents_0d, map_1d, Direction<1>::upper_xi());

  CHECK(normal_1d_upper.get(0) == DataVector(1, 0.2));

  check(ElementMap<2, TargetFrame>(
            ElementId<2>(0),
            make_coordinate_map_base<Frame::Logical, TargetFrame>(
                CoordinateMaps::Rotation<2>(atan2(4., 3.)))),
        {{{{0.6, 0.8}}, {{-0.8, 0.6}}}});

  check(ElementMap<3, TargetFrame>(
            ElementId<3>(0),
            make_coordinate_map_base<Frame::Logical, TargetFrame>(
                CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                               CoordinateMaps::Rotation<2>>(
                    {-1., 1., 2., 7.},
                    CoordinateMaps::Rotation<2>(atan2(4., 3.))))),
        {{{{0.4, 0., 0.}}, {{0., 0.6, 0.8}}, {{0., -0.8, 0.6}}}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FaceNormal.ElementMap", "[Unit][Domain]") {
  test_face_normal_element_map<Frame::Inertial>();
  test_face_normal_element_map<Frame::Grid>();
}

namespace {
struct Directions : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Directions";
  using type = std::unordered_set<Direction<2>>;
};
}  // namespace
SPECTRE_TEST_CASE("Unit.Domain.FaceNormal.ComputeItem", "[Unit][Domain]") {
  const auto box = db::create<
      db::AddSimpleTags<Directions, Tags::Extents<2>, Tags::ElementMap<2>>,
      db::AddComputeTags<
          Tags::Interface<Directions, Tags::Direction<2>>,
          Tags::Interface<Directions, Tags::Extents<1>>,
          Tags::Interface<Directions, Tags::UnnormalizedFaceNormal<2>>>>(
      std::unordered_set<Direction<2>>{Direction<2>::upper_xi(),
                                       Direction<2>::lower_eta()},
      Index<2>{{{2, 2}}},
      ElementMap<2, Frame::Inertial>(
          ElementId<2>(0),
          make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              CoordinateMaps::Rotation<2>(atan2(4., 3.)))));

  std::unordered_map<Direction<2>, tnsr::i<DataVector, 2>> expected;
  expected[Direction<2>::upper_xi()] =
      tnsr::i<DataVector, 2>{{{{0.6, 0.6}, {0.8, 0.8}}}};
  expected[Direction<2>::lower_eta()] =
      tnsr::i<DataVector, 2>{{{{0.8, 0.8}, {-0.6, -0.6}}}};

  CHECK_ITERABLE_APPROX(
      (get<Tags::Interface<Directions, Tags::UnnormalizedFaceNormal<2>>>(box)),
      expected);
}
