// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Domain.

#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Frame {
struct BlockLogical;
}  // namespace Frame
namespace PUP {
class er;
}  // namespace PUP
/// \cond
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
}  // namespace domain
/// \endcond

/*!
 * \brief Holds entities related to the computational domain.
 */
namespace domain {}

/*!
 *  \ingroup ComputationalDomainGroup
 *  \brief A wrapper around a vector of Blocks that represent the computational
 * domain.
 */
template <size_t VolumeDim>
class Domain {
 public:
  explicit Domain(std::vector<Block<VolumeDim>> blocks);

  /*!
   * \brief Create a Domain using CoordinateMaps to encode the Orientations.
   * This constructor does not support periodic boundary conditions.
   *
   * \details A constructor that does not require the user to provide a corner
   * numbering scheme. Constructs a global corner numbering for each pair
   * of abutting Blocks from their maps alone. The numbering is used to
   * set up the corresponding Orientation, and then is discarded; the
   * next pair of blocks uses a new global corner numbering, and so on,
   * until all pairs of abutting Blocks have had their Orientations
   * determined. For more information on setting up domains, see the
   * [domain creation tutorial](\ref tutorial_domain_creation).
   *
   * \param maps The BlockLogical -> Inertial coordinate map for each block.
   * \param excision_spheres Any ExcisionSphere%s in the domain.
   * \param block_names A human-readable name for every block, or empty if no
   * block names have been chosen (yet).
   * \param block_groups Labels to refer to groups of blocks. The groups can
   * overlap, and they don't have to cover all blocks in the domain. The groups
   * can be used to refer to multiple blocks at once.
   */
  explicit Domain(
      std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::BlockLogical, Frame::Inertial, VolumeDim>>>
          maps,
      std::unordered_map<std::string, ExcisionSphere<VolumeDim>>
          excision_spheres = {},
      std::vector<std::string> block_names = {},
      std::unordered_map<std::string, std::unordered_set<std::string>>
          block_groups = {});

  /*!
   * \brief Create a Domain using a corner numbering scheme to encode the
   * Orientations
   *
   * \see [domain creation tutorial](@ref tutorial_domain_creation)
   *
   * \param maps The BlockLogical -> Inertial coordinate map for each block.
   * \param corners_of_all_blocks The corner numbering for each block's corners
   * according to the global corner number scheme. The details of the corner
   * numbering scheme are described in the
   * [tutorial](@ref tutorial_orientations).
   * \param identifications Used to impose periodic boundary conditions on the
   * domain. To identify faces, `identifications` should contain the PairOfFaces
   * containing the corners of each pair of faces that you wish to identify with
   * one another. The number of `identifications` must be even.
   * \param excision_spheres Any ExcisionSphere%s in the domain.
   * \param block_names A human-readable name for every block, or empty if no
   * block names have been chosen (yet).
   * \param block_groups Labels to refer to groups of blocks. The groups can
   * overlap, and they don't have to cover all blocks in the domain. The groups
   * can be used to refer to multiple blocks at once.
   */
  Domain(std::vector<std::unique_ptr<domain::CoordinateMapBase<
             Frame::BlockLogical, Frame::Inertial, VolumeDim>>>
             maps,
         const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
             corners_of_all_blocks,
         const std::vector<PairOfFaces>& identifications = {},
         std::unordered_map<std::string, ExcisionSphere<VolumeDim>>
             excision_spheres = {},
         std::vector<std::string> block_names = {},
         std::unordered_map<std::string, std::unordered_set<std::string>>
             block_groups = {});

  Domain() = default;
  ~Domain() = default;
  Domain(const Domain&) = delete;
  Domain(Domain&&) = default;
  Domain<VolumeDim>& operator=(const Domain<VolumeDim>&) = delete;
  Domain<VolumeDim>& operator=(Domain<VolumeDim>&&) = default;

  void inject_time_dependent_map_for_block(
      size_t block_id,
      std::unique_ptr<
          domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
          moving_mesh_grid_to_inertial_map,
      std::unique_ptr<
          domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, VolumeDim>>
          moving_mesh_grid_to_distorted_map = nullptr,
      std::unique_ptr<domain::CoordinateMapBase<Frame::Distorted,
                                                Frame::Inertial, VolumeDim>>
          moving_mesh_distorted_to_inertial_map = nullptr);

  const std::vector<Block<VolumeDim>>& blocks() const { return blocks_; }

  bool is_time_dependent() const;

  const std::unordered_map<std::string, ExcisionSphere<VolumeDim>>&
  excision_spheres() const {
    return excision_spheres_;
  }

  /// Labels to refer to groups of blocks. The groups can overlap, and they
  /// don't have to cover all blocks in the domain. The groups can be used to
  /// refer to multiple blocks at once.
  const std::unordered_map<std::string, std::unordered_set<std::string>>&
  block_groups() const {
    return block_groups_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  std::vector<Block<VolumeDim>> blocks_{};
  std::unordered_map<std::string, ExcisionSphere<VolumeDim>>
      excision_spheres_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};
};

template <size_t VolumeDim>
bool operator==(const Domain<VolumeDim>& lhs, const Domain<VolumeDim>& rhs);

template <size_t VolumeDim>
bool operator!=(const Domain<VolumeDim>& lhs, const Domain<VolumeDim>& rhs);

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Domain<VolumeDim>& d);
