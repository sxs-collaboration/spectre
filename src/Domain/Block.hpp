// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Block.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <unordered_set>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"  // IWYU pragma: keep
#include "Domain/Structure/BlockNeighbor.hpp"       // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"           // IWYU pragma: keep
#include "Domain/Structure/DirectionMap.hpp"

/// \cond
namespace Frame {
struct Logical;
struct Inertial;
}  // namespace Frame
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup ComputationalDomainGroup
/// A Block<VolumeDim> is a region of a VolumeDim-dimensional computational
/// domain that defines the root node of a tree which is used to construct the
/// Elements that cover a region of the computational domain.
///
/// Each codimension 1 boundary of a Block<VolumeDim> is either an external
/// boundary or identical to a boundary of one other Block.
///
/// A Block has logical coordinates that go from -1 to +1 in each
/// dimension.  The global coordinates are obtained from the logical
/// coordinates from the Coordinatemap:  CoordinateMap::operator() takes
/// Points in the Logical Frame (i.e., logical coordinates) and
/// returns Points in the Inertial Frame (i.e., global coordinates).
template <size_t VolumeDim>
class Block {
 public:
  /// \param stationary_map the CoordinateMap.
  /// \param id a unique ID.
  /// \param neighbors info about the Blocks that share a codimension 1
  /// boundary with this Block.
  /// \param external_boundary_conditions the boundary conditions to be applied
  ///        at external boundaries. Can be either empty or one per each
  ///        external boundary
  Block(std::unique_ptr<domain::CoordinateMapBase<
            Frame::Logical, Frame::Inertial, VolumeDim>>&& stationary_map,
        size_t id, DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>> neighbors,
        DirectionMap<
            VolumeDim,
            std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
            external_boundary_conditions = {}) noexcept;

  Block() = default;
  ~Block() = default;
  Block(const Block&) = delete;
  Block(Block&&) = default;
  Block& operator=(const Block&) = delete;
  Block& operator=(Block&&) = default;

  /// \brief The map used when the coordinate map is time-independent.
  ///
  /// \see is_time_dependent()
  const domain::CoordinateMapBase<Frame::Logical, Frame::Inertial, VolumeDim>&
  stationary_map() const noexcept;

  /// \brief The map going from the block logical frame to the last time
  /// independent frame. Only used when the coordinate map is time-dependent.
  ///
  /// \see is_time_dependent() moving_mesh_grid_to_inertial_map()
  const domain::CoordinateMapBase<Frame::Logical, Frame::Grid, VolumeDim>&
  moving_mesh_logical_to_grid_map() const noexcept;

  /// \brief The map going from the last time independent frame to the frame in
  /// which the equations are solved. Only used when the coordinate map is
  /// time-dependent.
  ///
  /// \see is_time_dependent() moving_mesh_logical_to_grid_map()
  const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>&
  moving_mesh_grid_to_inertial_map() const noexcept;

  /// \brief Returns `true` if the block has time-dependent maps.
  bool is_time_dependent() const noexcept { return stationary_map_ == nullptr; }

  /// \brief Given a Block that has a time-independent map, injects the
  /// time-dependent map into the Block.
  void inject_time_dependent_map(
      std::unique_ptr<
          domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
          moving_mesh_inertial_map) noexcept;

  /// A unique identifier for the Block that is in the range
  /// [0, number_of_blocks -1] where number_of_blocks is the number
  /// of Blocks that cover the computational domain.
  size_t id() const noexcept { return id_; }

  /// Information about the neighboring Blocks.
  const DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>>& neighbors() const
      noexcept {
    return neighbors_;
  }

  /// The directions of the faces of the Block that are external boundaries.
  const std::unordered_set<Direction<VolumeDim>>& external_boundaries() const
      noexcept {
    return external_boundaries_;
  }

  /// The boundary conditions to apply at the external boundaries of the Block.
  const DirectionMap<
      VolumeDim,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>&
  external_boundary_conditions() const noexcept {
    return external_boundary_conditions_;
  }

  /// Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  template <size_t LocalVolumeDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Block<LocalVolumeDim>& lhs,
                         const Block<LocalVolumeDim>& rhs) noexcept;

  std::unique_ptr<
      domain::CoordinateMapBase<Frame::Logical, Frame::Inertial, VolumeDim>>
      stationary_map_{nullptr};
  std::unique_ptr<
      domain::CoordinateMapBase<Frame::Logical, Frame::Grid, VolumeDim>>
      moving_mesh_grid_map_{nullptr};
  std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
      moving_mesh_inertial_map_{nullptr};

  size_t id_{0};
  DirectionMap<VolumeDim, BlockNeighbor<VolumeDim>> neighbors_;
  std::unordered_set<Direction<VolumeDim>> external_boundaries_;
  DirectionMap<VolumeDim,
               std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
      external_boundary_conditions_;
};

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Block<VolumeDim>& block) noexcept;

template <size_t VolumeDim>
bool operator!=(const Block<VolumeDim>& lhs,
                const Block<VolumeDim>& rhs) noexcept;
