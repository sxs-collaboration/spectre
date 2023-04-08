// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/GhostData.hpp"

#include <cstddef>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace evolution::dg::subcell {
GhostData::GhostData(const size_t number_of_buffers)
    : number_of_buffers_(number_of_buffers) {
  if (number_of_buffers_ == 0) {
    ERROR("The GhostData class must be constructed with at least one buffer.");
  }

  local_ghost_data_.resize(number_of_buffers_);
  neighbor_ghost_data_for_reconstruction_.resize(number_of_buffers_);
  buffer_index_ = 0;
}

void GhostData::next_buffer() {
  buffer_index_ =
      buffer_index_ + 1 == number_of_buffers_ ? 0 : buffer_index_ + 1;
}

size_t GhostData::current_buffer_index() const { return buffer_index_; }

size_t GhostData::total_number_of_buffers() const { return number_of_buffers_; }

DataVector& GhostData::local_ghost_data() {
  return local_ghost_data_[buffer_index_];
}

const DataVector& GhostData::local_ghost_data() const {
  return local_ghost_data_[buffer_index_];
}

DataVector& GhostData::neighbor_ghost_data_for_reconstruction() {
  return neighbor_ghost_data_for_reconstruction_[buffer_index_];
}

const DataVector& GhostData::neighbor_ghost_data_for_reconstruction() const {
  return neighbor_ghost_data_for_reconstruction_[buffer_index_];
}

void GhostData::pup(PUP::er& p) {
  p | number_of_buffers_;
  p | buffer_index_;
  p | local_ghost_data_;
  // Once Charm++ messages are implemented, this will most likely contain
  // non-owning DataVectors which we will have to account for
  p | neighbor_ghost_data_for_reconstruction_;
}

bool operator==(const GhostData& lhs, const GhostData& rhs) {
  return lhs.number_of_buffers_ == rhs.number_of_buffers_ and
         lhs.buffer_index_ == rhs.buffer_index_ and
         lhs.local_ghost_data_ == rhs.local_ghost_data_ and
         lhs.neighbor_ghost_data_for_reconstruction_ ==
             rhs.neighbor_ghost_data_for_reconstruction_;
}

bool operator!=(const GhostData& lhs, const GhostData& rhs) {
  return not(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const GhostData& ghost_data) {
  using ::operator<<;
  os << "LocalGhostData: " << ghost_data.local_ghost_data() << "\n";
  os << "NeighborGhostDataForReconstruction: "
     << ghost_data.neighbor_ghost_data_for_reconstruction() << "\n";
  return os;
}
}  // namespace evolution::dg::subcell
