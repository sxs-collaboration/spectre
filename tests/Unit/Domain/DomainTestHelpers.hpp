// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/rational.hpp>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Side.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
template <size_t VolumeDim, typename TargetFrame>
class Block;
template <size_t VolumeDim>
class BlockNeighbor;
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim, typename TargetFrame>
class Domain;
template <size_t VolumeDim>
class ElementId;
namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame
/// \endcond

// Iterates over the logical corners of a VolumeDim-dimensional cube.
template <size_t VolumeDim>
class VolumeCornerIterator {
 public:
  VolumeCornerIterator() noexcept = default;
  explicit VolumeCornerIterator(size_t index) noexcept : index_(index) {}
  void operator++() noexcept {
    ++index_;
    for (size_t i = 0; i < VolumeDim; i++) {
      gsl::at(coords_of_corner_, i) = 2.0 * get_nth_bit(index_, i) - 1.0;
      gsl::at(array_sides_, i) =
          2 * get_nth_bit(index_, i) - 1 == 1 ? Side::Upper : Side::Lower;
    }
  }
  explicit operator bool() const noexcept {
    return index_ < two_to_the(VolumeDim);
  }
  const std::array<Side, VolumeDim>& operator()() const noexcept {
    return array_sides_;
  }
  const std::array<Side, VolumeDim>& operator*() const noexcept {
    return array_sides_;
  }
  const std::array<double, VolumeDim>& coords_of_corner() const noexcept {
    return coords_of_corner_;
  }

 private:
  size_t index_ = 0;
  std::array<Side, VolumeDim> array_sides_ = make_array<VolumeDim>(Side::Lower);
  std::array<double, VolumeDim> coords_of_corner_ = make_array<VolumeDim>(-1.0);
};

// Test that the Blocks in the Domain are constructed correctly.
template <size_t VolumeDim>
void test_domain_construction(
    const Domain<VolumeDim, Frame::Inertial>& domain,
    const std::vector<
        std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<VolumeDim>>>&
        expected_external_boundaries,
    const std::vector<std::unique_ptr<
        CoordinateMapBase<Frame::Logical, Frame::Inertial, VolumeDim>>>&
        expected_maps) noexcept;

// Test that two neighboring Blocks abut each other.
template <size_t VolumeDim, typename TargetFrame>
void test_physical_separation(
    const std::vector<Block<VolumeDim, TargetFrame>>& blocks) noexcept;

// Fraction of the logical volume of a block covered by an element
// The sum of this over all the elements of a block should be one
template <size_t VolumeDim>
boost::rational<size_t> fraction_of_block_volume(
    const ElementId<VolumeDim>& element_id) noexcept;

// Test that the Elements of the initial domain are connected and cover the
// computational domain, as well as that neighboring Elements  are at the same
// refinement level.
template <size_t VolumeDim>
void test_initial_domain(const Domain<VolumeDim, Frame::Inertial>& domain,
                         const std::vector<std::array<size_t, VolumeDim>>&
                             initial_refinement_levels) noexcept;
