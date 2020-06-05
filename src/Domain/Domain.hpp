// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Domain.

#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <vector>

#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/DomainHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Frame {
struct Logical;
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
 *  \ingroup ComputationalDomainGroup
 *  \brief A wrapper around a vector of Blocks that represent the computational
 * domain.
 */
template <size_t VolumeDim>
class Domain {
 public:
  explicit Domain(std::vector<Block<VolumeDim>> blocks) noexcept;

  /*!
   * Create a Domain using CoordinateMaps to encode the Orientations.
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
   */
  Domain(std::vector<std::unique_ptr<domain::CoordinateMapBase<
             Frame::Logical, Frame::Inertial, VolumeDim>>>
             maps) noexcept;

  /*!
   * Create a Domain using a corner numbering scheme to encode the Orientations,
   * with an optional parameter that encodes periodic boundary conditions.
   *
   * \details Each element of `corners_of_all_blocks` contains the corner
   * numbering of that block's corners according to the global corner number
   * scheme. The details of the corner numbering scheme are described in the
   * [tutorial](@ref tutorial_orientations). `identifications` is for imposing
   * periodic boundary conditions on the domain. To identify faces,
   * `identifications` should contain the PairOfFaces containing the corners of
   * each pair of faces that you wish to identify with one another. For more
   * information on setting up domains, see the
   * [domain creation tutorial](@ref tutorial_domain_creation).
   *
   * \requires `maps.size() == corners_of_all_blocks.size()`, and
   * `identifications.size()` is even.
   */
  Domain(std::vector<std::unique_ptr<domain::CoordinateMapBase<
             Frame::Logical, Frame::Inertial, VolumeDim>>>
             maps,
         const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
             corners_of_all_blocks,
         const std::vector<PairOfFaces>& identifications = {}) noexcept;

  Domain() noexcept = default;
  ~Domain() = default;
  Domain(const Domain&) = delete;
  Domain(Domain&&) = default;
  Domain<VolumeDim>& operator=(const Domain<VolumeDim>&) = delete;
  Domain<VolumeDim>& operator=(Domain<VolumeDim>&&) = default;

  void inject_time_dependent_map_for_block(
      size_t block_id,
      std::unique_ptr<
          domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>>
          moving_mesh_inertial_map) noexcept;

  const std::vector<Block<VolumeDim>>& blocks() const noexcept {
    return blocks_;
  }

  //clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  std::vector<Block<VolumeDim>> blocks_{};
};

template <size_t VolumeDim>
bool operator==(const Domain<VolumeDim>& lhs,
                const Domain<VolumeDim>& rhs) noexcept;

template <size_t VolumeDim>
bool operator!=(const Domain<VolumeDim>& lhs,
                const Domain<VolumeDim>& rhs) noexcept;

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Domain<VolumeDim>& d) noexcept;
