// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/rational.hpp>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"

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

// Fraction of the logical volume of a block covered by an element
// The sum of this over all the elements of a block should be one
template <size_t VolumeDim>
boost::rational<size_t> fraction_of_block_volume(
    const ElementId<VolumeDim>& element_id) noexcept;
