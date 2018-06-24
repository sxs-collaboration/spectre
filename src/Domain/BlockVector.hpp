// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <ostream>
#include <vector>

#include "Domain/BlockId.hpp"

namespace domain {
/*!
 * \ingroup ComputationalDomainGroup
 * \brief A vector that is indexed by `BlockId`.
 */
template <class T, class Allocator = std::allocator<T>>
class BlockVector : private std::vector<T, Allocator> {
 public:
  // Member types
  using value_type = typename std::vector<T, Allocator>::value_type;
  using allocator_type = typename std::vector<T, Allocator>::allocator_type;
  using size_type = typename std::vector<T, Allocator>::size_type;
  using difference_type = typename std::vector<T, Allocator>::difference_type;
  using reference = typename std::vector<T, Allocator>::reference;
  using const_reference = typename std::vector<T, Allocator>::const_reference;
  using pointer = typename std::vector<T, Allocator>::pointer;
  using const_pointer = typename std::vector<T, Allocator>::const_pointer;
  using iterator = typename std::vector<T, Allocator>::iterator;
  using const_iterator = typename std::vector<T, Allocator>::const_iterator;
  using reverse_iterator = typename std::vector<T, Allocator>::reverse_iterator;
  using const_reverse_iterator =
      typename std::vector<T, Allocator>::const_reverse_iterator;

  // Member functions
  using std::vector<T, Allocator>::vector;
  using std::vector<T, Allocator>::operator=;
  using std::vector<T, Allocator>::assign;
  using std::vector<T, Allocator>::get_allocator;

  // Element access
  reference at(const BlockId& pos) noexcept {
    return static_cast<std::vector<T, Allocator>&>(*this).at(pos.get_index());
  }
  const_reference at(const BlockId& pos) const noexcept {
    return static_cast<const std::vector<T, Allocator>&>(*this).at(
        pos.get_index());
  }
  reference operator[](const BlockId& pos) noexcept {
    return static_cast<std::vector<T, Allocator>&>(*this)[pos.get_index()];
  }
  const_reference operator[](const BlockId& pos) const noexcept {
    return static_cast<const std::vector<T, Allocator>&>(
        *this)[pos.get_index()];
  }
  using std::vector<T, Allocator>::front;
  using std::vector<T, Allocator>::back;
  using std::vector<T, Allocator>::data;

  // Iterators
  using std::vector<T, Allocator>::begin;
  using std::vector<T, Allocator>::cbegin;
  using std::vector<T, Allocator>::end;
  using std::vector<T, Allocator>::cend;

  // Capacity
  using std::vector<T, Allocator>::empty;
  using std::vector<T, Allocator>::size;
  using std::vector<T, Allocator>::max_size;
  using std::vector<T, Allocator>::reserve;
  using std::vector<T, Allocator>::capacity;
  using std::vector<T, Allocator>::shrink_to_fit;

  // Modifiers
  using std::vector<T, Allocator>::clear;
  using std::vector<T, Allocator>::push_back;
  using std::vector<T, Allocator>::emplace_back;
  using std::vector<T, Allocator>::pop_back;
  using std::vector<T, Allocator>::resize;
  using std::vector<T, Allocator>::swap;

  // Charm++ serialization
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept {
    p | static_cast<std::vector<T, Allocator>&>(*this);
  }

#define GENERATE_OP(OP)                                                        \
  template <class S, class SAllocator = std::allocator<S>>                     \
  friend bool operator OP(                                                     \
      const domain::BlockVector<S, SAllocator>& lhs,                           \
      const domain::BlockVector<S, SAllocator>& rhs) noexcept {                \
    return static_cast<const std::vector<S, SAllocator>&>(lhs) OP /* NOLINT */ \
        static_cast<const std::vector<S, SAllocator>&>(rhs);                   \
  }
  GENERATE_OP(==)
  GENERATE_OP(!=)
  GENERATE_OP(<)
  GENERATE_OP(<=)
  GENERATE_OP(>)
  GENERATE_OP(>=)
#undef GENERATE_OP

  template <class S, class SAllocator = std::allocator<T>>
  friend std::ostream& operator<<(
      std::ostream& os, const BlockVector<S, SAllocator>& block_vector) {
    // We intentionally don't use the std::vector stream operator since
    // including that would suck in almost the entire STL.
    const auto& base =
        static_cast<const std::vector<S, SAllocator>&>(block_vector);
    os << '(';
    for (size_t i = 0; i < base.size(); ++i) {
      os << (i > 0 ? "," : "") << base[i];
    }
    return os << ')';
  }
};
}  // namespace domain
