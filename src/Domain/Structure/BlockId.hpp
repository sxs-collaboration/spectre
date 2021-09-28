// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>
#include <pup.h>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
/*!
 * \ingroup ComputationalDomainGroup
 * \brief Index a block of the computational domain.
 */
class BlockId {
 public:
  BlockId() = default;
  explicit BlockId(size_t id) : id_(id) {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | id_; }

  size_t get_index() const { return id_; }

  BlockId& operator++() {
    ++id_;
    return *this;
  }

  BlockId& operator--() {
    --id_;
    return *this;
  }

  // NOLINTNEXTLINE(cert-dcl21-cpp) returned object doesn't need to be const
  BlockId operator++(int) {
    BlockId temp = *this;
    id_++;
    return temp;
  }

  // NOLINTNEXTLINE(cert-dcl21-cpp) returned object doesn't need to be const
  BlockId operator--(int) {
    BlockId temp = *this;
    id_--;
    return temp;
  }

 private:
  size_t id_{0};
};

inline bool operator==(const BlockId& lhs, const BlockId& rhs) {
  return lhs.get_index() == rhs.get_index();
}

inline bool operator!=(const BlockId& lhs, const BlockId& rhs) {
  return not(lhs == rhs);
}

inline std::ostream& operator<<(std::ostream& os, const BlockId& block_id) {
  return os << '[' << block_id.get_index() << ']';
}

}  // namespace domain
