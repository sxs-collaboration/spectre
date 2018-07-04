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

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
namespace domain {
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
}  // namespace domain
/// \endcond

// Test that the domain::Blocks in the domain::Domain are constructed correctly.
template <size_t VolumeDim, typename Fr = Frame::Inertial>
void test_domain_construction(
    const domain::Domain<VolumeDim, Fr>& domain,
    const std::vector<std::unordered_map<domain::Direction<VolumeDim>,
                                         domain::BlockNeighbor<VolumeDim>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<domain::Direction<VolumeDim>>>&
        expected_external_boundaries,
    const std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Logical, Fr, VolumeDim>>>&
        expected_maps) noexcept;

// Test that two neighboring domain::Blocks abut each other.
template <size_t VolumeDim, typename Fr = Frame::Inertial>
void test_physical_separation(
    const std::vector<domain::Block<VolumeDim, Fr>>& blocks) noexcept;

// Fraction of the logical volume of a block covered by an element
// The sum of this over all the elements of a block should be one
template <size_t VolumeDim>
boost::rational<size_t> fraction_of_block_volume(
    const domain::ElementId<VolumeDim>& element_id) noexcept;

// Test that the domain::Elements of the initial domain are connected and cover
// the computational domain, as well as that neighboring domain::Elements  are
// at the same refinement level.
template <size_t VolumeDim, typename Fr = Frame::Inertial>
void test_initial_domain(const domain::Domain<VolumeDim, Fr>& domain,
                         const std::vector<std::array<size_t, VolumeDim>>&
                             initial_refinement_levels) noexcept;

// Euclidean basis vector along the given `domain::Direction` and in the given
// `Frame`.
template <size_t SpatialDim, typename Fr = Frame::Inertial>
tnsr::i<DataVector, SpatialDim, Fr> euclidean_basis_vector(
    const domain::Direction<SpatialDim>& direction,
    const DataVector& used_for_size) noexcept;
