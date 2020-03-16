// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
void test_1d() noexcept {
  INFO("1d");
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
      domain::CoordinateMaps::Affine(-1.0, 1.0, 0.3, 1.2));
  const ElementId<1> element_id(0,
                                std::array<SegmentId, 1>({{SegmentId(2, 3)}}));
  const ElementMap<1, Frame::Inertial> element_map(element_id, std::move(map));
  const auto size = size_of_element(element_map);
  // for this affine map, expected size = width of block / number of elements
  const auto size_expected = make_array<1>(0.225);
  CHECK_ITERABLE_APPROX(size, size_expected);
}

void test_2d() noexcept {
  INFO("2d");
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
      Affine2D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2)},
      domain::CoordinateMaps::DiscreteRotation<2>(
          OrientationMap<2>{std::array<Direction<2>, 2>{
              {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}));
  const ElementId<2> element_id(
      0, std::array<SegmentId, 2>({{SegmentId(1, 1), SegmentId(2, 0)}}));
  const ElementMap<2, Frame::Inertial> element_map(element_id, std::move(map));
  const auto size = size_of_element(element_map);
  const auto size_expected = make_array(0.05, 0.425);
  CHECK_ITERABLE_APPROX(size, size_expected);
}

void test_3d() noexcept {
  INFO("3d");
  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
      Affine3D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2),
               Affine(-1.0, 1.0, 12.0, 12.5)},
      domain::CoordinateMaps::Rotation<3>(0.7, 2.3, -0.4));
  const ElementId<3> element_id(
      0, std::array<SegmentId, 3>(
             {{SegmentId(3, 5), SegmentId(1, 0), SegmentId(2, 3)}}));
  const ElementMap<3, Frame::Inertial> element_map(element_id, std::move(map));
  const auto size = size_of_element(element_map);
  const auto size_expected = make_array(0.0125, 0.85, 0.125);
  CHECK_ITERABLE_APPROX(size, size_expected);
}

void test_compute_tag() noexcept {
  INFO("compute tag");
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  auto map = domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
      Affine2D{Affine(-1.0, 1.0, 0.3, 0.4), Affine(-1.0, 1.0, -0.5, 1.2)},
      domain::CoordinateMaps::DiscreteRotation<2>(
          OrientationMap<2>{std::array<Direction<2>, 2>{
              {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}));
  const ElementId<2> element_id(
      0, std::array<SegmentId, 2>({{SegmentId(1, 1), SegmentId(2, 0)}}));
  ElementMap<2, Frame::Inertial> element_map(element_id, std::move(map));
  const auto box = db::create<
      db::AddSimpleTags<domain::Tags::ElementMap<2, Frame::Inertial>>,
      db::AddComputeTags<domain::Tags::SizeOfElement<2>>>(
      std::move(element_map));

  const auto size_compute_item = db::get<domain::Tags::SizeOfElement<2>>(box);
  const auto size_expected = make_array(0.05, 0.425);
  CHECK_ITERABLE_APPROX(size_compute_item, size_expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.SizeOfElement", "[Domain][Unit]") {
  test_1d();
  test_2d();
  test_3d();
  test_compute_tag();
}
