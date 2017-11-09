// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

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
void test_domain_construction(
    const Domain<VolumeDim, Frame::Inertial>& domain,
    const std::vector<
        std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<VolumeDim>>>&
        expected_external_boundaries,
    const std::vector<std::unique_ptr<
        CoordinateMapBase<Frame::Logical, Frame::Inertial, VolumeDim>>>&
        expected_maps);
