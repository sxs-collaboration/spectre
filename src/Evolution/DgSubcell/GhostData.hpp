// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <pup.h>
#include <vector>

#include "DataStructures/DataVector.hpp"

namespace evolution::dg::subcell {
/*!
 * Ghost data used on the subcell grid for reconstruction
 *
 * This class holds both the local ghost data on the local subcell mesh for a
 * given direction, as well as the neighbor's ghost data (on the neighbor's
 * mesh) in that same direction. This class is similar to
 * `evolution::dg::MortarData` in the sense that it holds both local and
 * neighbor data in a direction. However, it differs because the local ghost
 * data is not used in our own calculation when reconstructing the solution at
 * the face between the elements. This is because we already have our own data
 * on our own FD grid. Only the neighbor ghost data is used to reconstruct the
 * solution on the face.
 *
 * With Charm++ messages, storing the local ghost data is necessary because it
 * must live somewhere so we can send a pointer to our neighbor.
 */
class GhostData {
 public:
  GhostData(size_t number_of_buffers = 1);

  /// Move to the next internal mortar buffer
  void next_buffer();

  /// Return the current internal buffer index
  size_t current_buffer_index() const;

  /// Return the total number of buffers that this GhostData was constructed
  /// with
  size_t total_number_of_buffers() const;

  /// @{
  /// The local ghost data for in the current buffer
  ///
  /// The non-const reference function can be used to edit the data in-place
  DataVector& local_ghost_data();

  const DataVector& local_ghost_data() const;
  /// @}

  /// @{
  /// The ghost data from our neighbor which is meant to be used in
  /// reconstruction for in the current buffer.
  ///
  /// The non-const reference function can be used to edit the data in-place
  DataVector& neighbor_ghost_data_for_reconstruction();

  const DataVector& neighbor_ghost_data_for_reconstruction() const;
  /// @}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  // NOLINTNEXTLINE
  friend bool operator==(const GhostData& lhs, const GhostData& rhs);

  size_t number_of_buffers_{1};
  size_t buffer_index_{0};
  std::vector<DataVector> local_ghost_data_{};
  std::vector<DataVector> neighbor_ghost_data_for_reconstruction_{};
};

bool operator!=(const GhostData& lhs, const GhostData& rhs);

std::ostream& operator<<(std::ostream& os, const GhostData& ghost_data);
}  // namespace evolution::dg::subcell
