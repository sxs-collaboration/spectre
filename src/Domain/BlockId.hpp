// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>
#include <pup.h>

namespace PUP {
class er;
}  // namespace PUP

namespace domain {
/*!
 * \ingroup ComputationalDomainGroup
 * \brief Index a block of the computational domain.
 */
class BlockId {
 public:
  BlockId() = default;
  explicit BlockId(size_t id) noexcept : id_(id) {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept { p | id_; }

  size_t get_index() const noexcept { return id_; }

  BlockId& operator++() noexcept {
    ++id_;
    return *this;
  }

  BlockId& operator--() noexcept {
    --id_;
    return *this;
  }

  // NOLINTNEXTLINE(cert-dcl21-cpp) returned object doesn't need to be const
  BlockId operator++(int) noexcept {
    id_++;
    return *this;
  }

  // NOLINTNEXTLINE(cert-dcl21-cpp) returned object doesn't need to be const
  BlockId operator--(int) noexcept {
    id_--;
    return *this;
  }

 private:
  friend bool operator==(const BlockId& lhs, const BlockId& rhs) noexcept;
  friend std::ostream& operator<<(std::ostream& os, const BlockId& block_id);

  size_t id_{0};
};

inline bool operator==(const BlockId& lhs, const BlockId& rhs) noexcept {
  return lhs.id_ == rhs.id_;
}

inline bool operator!=(const BlockId& lhs, const BlockId& rhs) noexcept {
  return not(lhs == rhs);
}

inline std::ostream& operator<<(std::ostream& os, const BlockId& block_id) {
  return os << '[' << block_id.id_ << ']';
}

}  // namespace domain
