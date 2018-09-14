// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/TensorData.hpp"

#include <ostream>
#include <pup.h>
#include <pup_stl.h>

void TensorComponent::pup(PUP::er& p) noexcept {
  p | name;
  p | data;
}

std::ostream& operator<<(std::ostream& os, const TensorComponent& t) noexcept {
  return os << "(" << t.name << ", " << t.data << ")";
}

bool operator==(const TensorComponent& lhs,
                const TensorComponent& rhs) noexcept {
  return lhs.name == rhs.name and lhs.data == rhs.data;
}

bool operator!=(const TensorComponent& lhs,
                const TensorComponent& rhs) noexcept {
  return not(lhs == rhs);
}

void ExtentsAndTensorVolumeData::pup(PUP::er& p) noexcept {
  p | extents;
  p | tensor_components;
}
