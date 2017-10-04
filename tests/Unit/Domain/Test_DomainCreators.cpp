// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <algorithm>
#include <catch.hpp>

#include "Domain/Block.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
// Iterates over the logical corners of a VolumeDim-dimensional cube.
template <size_t VolumeDim>
class VolumeCornerIterator {
 public:
  VolumeCornerIterator() noexcept = default;
  void operator++() noexcept {
    ++index_;
    for (size_t i = 0; i < VolumeDim; i++) {
      corner_[i] = 2 * static_cast<int>(get_nth_bit(index_, i)) - 1;
    }
  }
  explicit operator bool() noexcept { return index_ < two_to_the(VolumeDim); }
  tnsr::I<double, VolumeDim, Frame::Logical> operator()() noexcept {
    return corner_;
  }
  tnsr::I<double, VolumeDim, Frame::Logical> operator*() noexcept {
    return corner_;
  }

 private:
  size_t index_ = 0;
  tnsr::I<double, VolumeDim, Frame::Logical> corner_ =
      make_array<VolumeDim>(-1);
};

// Iterates over the 2^(VolumeDim-1) logical corners of the face of a
// VolumeDim-dimensional cube in the given direction.
template <size_t VolumeDim>
class FaceCornerIterator {
 public:
  explicit FaceCornerIterator(Direction<VolumeDim> direction) noexcept;
  void operator++() noexcept {
    face_index_++;
    do {
      index_++;
    } while (get_nth_bit(index_, direction_.dimension()) ==
             (direction_.side() == Side::Upper ? 0 : 1));
    for (size_t i = 0; i < VolumeDim; ++i) {
      corner_[i] = 2 * static_cast<int>(get_nth_bit(index_, i)) - 1;
    }
  }
  explicit operator bool() noexcept { return index_ < two_to_the(VolumeDim); }
  tnsr::I<double, VolumeDim, Frame::Logical> operator()() noexcept {
    return corner_;
  }
  tnsr::I<double, VolumeDim, Frame::Logical> operator*() noexcept {
    return corner_;
  }

  // Returns the value used to construct the logical corner.
  size_t volume_index() noexcept { return index_; }
  // Returns the number of times operator++ has been called.
  size_t face_index() noexcept { return face_index_; }

 private:
  const Direction<VolumeDim> direction_;
  size_t index_;
  size_t face_index_ = 0;
  tnsr::I<double, VolumeDim, Frame::Logical> corner_;
};

template <size_t VolumeDim>
FaceCornerIterator<VolumeDim>::FaceCornerIterator(
    Direction<VolumeDim> direction) noexcept
    : direction_(std::move(direction)),
      index_(direction.side() == Side::Upper
                 ? two_to_the(direction_.dimension())
                 : 0) {
  for (size_t i = 0; i < VolumeDim; ++i) {
    corner_[i] = 2 * static_cast<int>(get_nth_bit(index_, i)) - 1;
  }
}

template <size_t VolumeDim, typename TargetFrame>
bool blocks_are_neighbors(
    const Block<VolumeDim, TargetFrame>& host_block,
    const Block<VolumeDim, TargetFrame>& neighbor_block) noexcept {
  for (const auto& neighbor : host_block.neighbors()) {
    if (neighbor.second.id() == neighbor_block.id()) {
      return true;
    }
  }
  return false;
}

// Finds the Orientation of a neighboring Block relative to a host Block.
template <size_t VolumeDim, typename TargetFrame>
Orientation<VolumeDim> find_neighbor_orientation(
    const Block<VolumeDim, TargetFrame>& host_block,
    const Block<VolumeDim, TargetFrame>& neighbor_block) noexcept {
  for (const auto& neighbor : host_block.neighbors()) {
    if (neighbor.second.id() == neighbor_block.id()) {
      return neighbor.second.orientation();
    }
  }
  ERROR("The Block `neighbor_block` is not a neighbor of `host_block`.");
}

// Finds the Direction to the neighboring Block relative to a host Block.
template <size_t VolumeDim, typename TargetFrame>
Direction<VolumeDim> find_direction_to_neighbor(
    const Block<VolumeDim, TargetFrame>& host_block,
    const Block<VolumeDim, TargetFrame>& neighbor_block) noexcept {
  for (const auto& neighbor : host_block.neighbors()) {
    if (neighbor.second.id() == neighbor_block.id()) {
      return neighbor.first;
    }
  }
  ERROR("The Block `neighbor_block` is not a neighbor of `host_block`.");
}

// Convert Point to Directions, for use with Orientation
template <size_t VolumeDim>
std::array<Direction<VolumeDim>, VolumeDim> get_orthant(
    const tnsr::I<double, VolumeDim, Frame::Logical>& point) noexcept {
  std::array<Direction<VolumeDim>, VolumeDim> result;
  for (size_t i = 0; i < VolumeDim; i++) {
    gsl::at(result, i) =
        Direction<VolumeDim>(i, point[i] >= 0 ? Side::Upper : Side::Lower);
  }
  return result;
}

// Convert Directions to Point, for use with CoordinateMap
template <size_t VolumeDim>
tnsr::I<double, VolumeDim, Frame::Logical> get_corner_of_orthant(
    const std::array<Direction<VolumeDim>, VolumeDim>& directions) noexcept {
  tnsr::I<double, VolumeDim, Frame::Logical> result{};
  for (size_t i = 0; i < VolumeDim; i++) {
    result[gsl::at(directions, i).dimension()] =
        gsl::at(directions, i).side() == Side::Upper ? 1.0 : -1.0;
  }
  return result;
}

// The relative Orientation between Blocks induces a map that takes
// Points in the host Block to Points in the neighbor Block.
template <size_t VolumeDim>
tnsr::I<double, VolumeDim, Frame::Logical> point_in_neighbor_frame(
    const Orientation<VolumeDim>& orientation,
    const tnsr::I<double, VolumeDim, Frame::Logical>& point) noexcept {
  auto point_get_orthant = get_orthant(point);
  std::for_each(point_get_orthant.begin(), point_get_orthant.end(),
                [&orientation](auto& direction) {
                  direction = orientation.mapped(direction);
                });
  return get_corner_of_orthant(point_get_orthant);
}

template <size_t VolumeDim, typename TargetFrame>
double physical_separation(
    const Block<VolumeDim, TargetFrame>& block1,
    const Block<VolumeDim, TargetFrame>& block2) noexcept {
  double max_separation = 0;
  // Find Direction to block2:
  const auto& direction = find_direction_to_neighbor(block1, block2);
  // Find Orientation relative to block2:
  const auto& orientation = find_neighbor_orientation(block1, block2);
  // Construct shared Points, in frame of block1:
  std::array<tnsr::I<double, VolumeDim, Frame::Logical>, VolumeDim>
      shared_points1;
  std::array<tnsr::I<double, VolumeDim, Frame::Logical>, VolumeDim>
      shared_points2;
  for (FaceCornerIterator<VolumeDim> fci(direction); fci; ++fci) {
    shared_points1[fci.face_index()] = fci();
  }
  // Construct shared Points, in frame of block2:
  for (FaceCornerIterator<VolumeDim> fci(direction.opposite()); fci; ++fci) {
    shared_points2[fci.face_index()] =
        point_in_neighbor_frame(orientation, fci());
  }
  // Obtain coordinate maps
  const auto& map1 = block1.coordinate_map();
  const auto& map2 = block2.coordinate_map();
  for (size_t i = 0; i < two_to_the(VolumeDim - 1); i++) {
    for (size_t j = 0; j < VolumeDim; j++) {
      max_separation = std::max(
          max_separation,
          abs(map1(shared_points1[i]).get(j) - map2(shared_points2[i]).get(j)));
    }
  }
  return max_separation;
}

template <size_t VolumeDim, typename TargetFrame>
void test_physical_separation(
    const std::vector<Block<VolumeDim, TargetFrame>>& blocks) noexcept {
  double tolerance = 1e-10;
  for (size_t i = 0; i < blocks.size() - 1; i++) {
    for (size_t j = i + 1; j < blocks.size(); j++) {
      if (blocks_are_neighbors(blocks[i], blocks[j])) {
        CHECK(physical_separation(blocks[i], blocks[j]) < tolerance);
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.DomainCreators.TestHelperFunctions",
                  "[Domain][Unit]") {
  Orientation<3> custom_orientation(std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::upper_zeta(),
       Direction<3>::lower_xi()}});
  BlockNeighbor<3> custom_neighbor(1, custom_orientation);

  tnsr::I<double, 3, Frame::Logical> test_point{{{1.0, 1.0, 1.0}}};
  tnsr::I<double, 3, Frame::Logical> expected{{{-1.0, -1.0, 1.0}}};
  CHECK(point_in_neighbor_frame<3>(custom_orientation, test_point) == expected);
}
