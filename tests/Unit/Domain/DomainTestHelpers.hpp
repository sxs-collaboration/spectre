// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/rational.hpp>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"

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
