// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/rational.hpp>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
template <size_t VolumeDim>
class Block;
template <size_t VolumeDim>
class BlockNeighbor;
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
class DataVector;
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim, typename T>
class DirectionMap;
template <size_t VolumeDim>
class Domain;
template <size_t VolumeDim>
class ElementId;
namespace domain::BoundaryConditions {
class BoundaryCondition;
}  // namespace domain::BoundaryConditions
/// \endcond

// Test that the Blocks in the Domain are constructed correctly.
//
// The boundary conditions test assumes that the
// `TestHelpers::domain::BoundaryConditions::TestBoundaryCondition` class is
// used as the concrete boundary condition.
template <size_t VolumeDim, typename TargetFrameGridOrInertial>
void test_domain_construction(
    const Domain<VolumeDim>& domain,
    const std::vector<DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>>&
        expected_block_neighbors,
    const std::vector<std::unordered_set<Direction<VolumeDim>>>&
        expected_external_boundaries,
    const std::vector<std::unique_ptr<domain::CoordinateMapBase<
        Frame::BlockLogical, TargetFrameGridOrInertial, VolumeDim>>>&
        expected_maps,
    double time = std::numeric_limits<double>::quiet_NaN(),
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = {},
    const std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>>&
        expected_grid_to_inertial_maps = {},
    const std::vector<DirectionMap<
        VolumeDim,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>&
        expected_boundary_conditions = {});

// Test that two neighboring Blocks abut each other.
template <size_t VolumeDim>
void test_physical_separation(
    const std::vector<Block<VolumeDim>>& blocks,
    double time = std::numeric_limits<double>::signaling_NaN(),
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = {}) noexcept;

// Given a vector of Blocks, tests that the determinant of the Jacobian is
// positive at all corners of those Blocks.
template <size_t VolumeDim>
void test_det_jac_positive(
    const std::vector<Block<VolumeDim>>& blocks,
    double time = std::numeric_limits<double>::signaling_NaN(),
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = {}) noexcept;

// Fraction of the logical volume of a block covered by an element
// The sum of this over all the elements of a block should be one
template <size_t VolumeDim>
boost::rational<size_t> fraction_of_block_volume(
    const ElementId<VolumeDim>& element_id) noexcept;

// Test that the Elements of the initial domain are connected and cover the
// computational domain, as well as that neighboring Elements' refinement
// levels do not differ too much.
template <size_t VolumeDim>
void test_initial_domain(const Domain<VolumeDim>& domain,
                         const std::vector<std::array<size_t, VolumeDim>>&
                             initial_refinement_levels) noexcept;

// Euclidean basis vector along the given `Direction` and in the given
// `Frame::Inertial` frame.
template <typename DataType, size_t SpatialDim>
tnsr::i<DataType, SpatialDim> euclidean_basis_vector(
    const Direction<SpatialDim>& direction,
    const DataType& used_for_size) noexcept;

template <typename DataType, size_t SpatialDim>
tnsr::i<DataType, SpatialDim> unit_basis_form(
    const Direction<SpatialDim>& direction,
    const tnsr::II<DataType, SpatialDim>& inv_spatial_metric) noexcept;
