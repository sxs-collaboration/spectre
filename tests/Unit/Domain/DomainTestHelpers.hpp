// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/rational.hpp>
#include <cstddef>
#include <memory>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
template <size_t VolumeDim, typename TargetFrame>
class Block;
template <size_t VolumeDim>
class BlockNeighbor;
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
}  // namespace domain
class DataVector;
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim, typename T>
class DirectionMap;
template <size_t VolumeDim, typename TargetFrame>
class Domain;
template <size_t VolumeDim>
class ElementId;
/// \endcond

// Test that the Blocks in the Domain are constructed correctly.
template <size_t VolumeDim, typename Fr = Frame::Inertial>
void test_domain_construction(
    const Domain<VolumeDim, Fr>& domain,
    const std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<VolumeDim>>>&
        expected_external_boundaries,
    const std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Logical, Fr, VolumeDim>>>&
        expected_maps) noexcept;

// Test that two neighboring Blocks abut each other.
template <size_t VolumeDim, typename Fr = Frame::Inertial>
void test_physical_separation(
    const std::vector<Block<VolumeDim, Fr>>& blocks) noexcept;

// Fraction of the logical volume of a block covered by an element
// The sum of this over all the elements of a block should be one
template <size_t VolumeDim>
boost::rational<size_t> fraction_of_block_volume(
    const ElementId<VolumeDim>& element_id) noexcept;

// Test that the Elements of the initial domain are connected and cover the
// computational domain, as well as that neighboring Elements  are at the same
// refinement level.
template <size_t VolumeDim, typename Fr = Frame::Inertial>
void test_initial_domain(const Domain<VolumeDim, Fr>& domain,
                         const std::vector<std::array<size_t, VolumeDim>>&
                             initial_refinement_levels) noexcept;

// Euclidean basis vector along the given `Direction` and in the given `Frame`.
template <size_t SpatialDim, typename Fr = Frame::Inertial>
tnsr::i<DataVector, SpatialDim, Fr> euclidean_basis_vector(
    const Direction<SpatialDim>& direction,
    const DataVector& used_for_size) noexcept;
