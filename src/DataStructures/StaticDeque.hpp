// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <pup.h>
#include <type_traits>

#include "Utilities/PrintHelpers.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StlBoilerplate.hpp"

namespace static_deque_detail {
// "The extent to which an implementation determines that a type
// cannot be an input iterator is unspecified, except that as a
// minimum integral types shall not qualify as input iterators."
//   -- N4659 26.2.1.17
template <typename U>
constexpr static bool is_input_iterator = not std::is_integral_v<U>;
}  // namespace static_deque_detail

/// A class implementing the std::deque interface with static storage.
///
/// Differences from std::deque:
/// * The size of the container cannot be increased above \p Capacity.
/// * Operations moving the entire container are \f$O(n)\f$ instead of
///   \f$O(1)\f$ in the size and invalidate all references and
///   iterators.
/// * Erasing elements from the front of the queue invalidates all
///   iterators (but not references).
///
/// This last point is not a fundamental limitation, but could be
/// corrected with a more complicated iterator implementation if the
/// standard behavior is found to be useful.
template <typename T, size_t Capacity>
class StaticDeque
    : public stl_boilerplate::RandomAccessSequence<StaticDeque<T, Capacity>,
                                                   T, true> {
 public:
  // Aliases needed in the class definition.  The rest can just be
  // inherited.
  using size_type = typename StaticDeque::size_type;
  using difference_type = typename StaticDeque::difference_type;
  using iterator = typename StaticDeque::iterator;
  using const_iterator = typename StaticDeque::const_iterator;

  StaticDeque() = default;
  explicit StaticDeque(size_type count);
  StaticDeque(size_type n, const T& value);
  template <
      typename InputIterator,
      Requires<static_deque_detail::is_input_iterator<InputIterator>> = nullptr>
  StaticDeque(InputIterator first, InputIterator last);
  StaticDeque(const StaticDeque& other);
  StaticDeque(StaticDeque&& other);
  StaticDeque(std::initializer_list<T> init);

  ~StaticDeque();

  StaticDeque& operator=(const StaticDeque& other);
  StaticDeque& operator=(StaticDeque&& other);
  StaticDeque& operator=(std::initializer_list<T> init);

  template <
      typename InputIterator,
      Requires<static_deque_detail::is_input_iterator<InputIterator>> = nullptr>
  void assign(InputIterator first, InputIterator last);
  void assign(size_type count, const T& value);
  void assign(std::initializer_list<T> init);

  size_type size() const { return size_; }
  size_type max_size() const { return Capacity; }
  void resize(size_type count);
  void resize(size_type count, const T& value);
  void shrink_to_fit() {}

  T& operator[](size_type n) { return *entry_address(n); }
  const T& operator[](size_type n) const { return *entry_address(n); }

  template <typename... Args>
  T& emplace_front(Args&&... args);
  template <typename... Args>
  T& emplace_back(Args&&... args);
  template <typename... Args>
  iterator emplace(const_iterator position, Args&&... args);

  void push_front(const T& x) { emplace_front(x); }
  void push_front(T&& x) { emplace_front(std::move(x)); }
  void push_back(const T& x) { emplace_back(x); }
  void push_back(T&& x) { emplace_back(std::move(x)); }

  iterator insert(const_iterator position, const T& x);
  iterator insert(const_iterator position, T&& x);
  iterator insert(const_iterator position, size_type n, const T& x);
  template <
      typename InputIterator,
      Requires<static_deque_detail::is_input_iterator<InputIterator>> = nullptr>
  iterator insert(const_iterator position, InputIterator first,
                  InputIterator last);
  iterator insert(const_iterator position, std::initializer_list<T> init);

  void pop_front();
  void pop_back();

  iterator erase(const_iterator position);
  iterator erase(const_iterator first, const_iterator last);

  void swap(StaticDeque& other);
  void clear();

  void pup(PUP::er& p);

 private:
  // Operations on the middle of the deque are supposed to shift as
  // few elements as possible.
  bool should_operate_on_back(const const_iterator& first,
                              const const_iterator& last) const {
    return (this->end() - last) <= (first - this->begin());
  }

  // Index into the internal array of T of logical index n
  size_type internal_index(size_type n) const {
    auto index = start_position_ + n;
    if (index >= Capacity) {
      index -= Capacity;
    }
    ASSERT(index < static_cast<difference_type>(Capacity),
           "Calculated out-of-range index: " << index);
    return index;
  }

  const T* entry_address(size_type n) const {
    return reinterpret_cast<const T*>(data_) + internal_index(n);
  }

  T* entry_address(size_type n) {
    return reinterpret_cast<T*>(data_) + internal_index(n);
  }

  // This array contains the memory used to store the elements.  It is
  // treated as an array of T, except that some elements may be
  // missing, i.e., be segments of memory not containing objects.
  // Elements are created using placement new and removed by calling
  // their destructors.
  //
  // Logically, the array is considered to be a periodic structure.
  // The existing elements are stored contiguously in the logical
  // structure, which may correspond to a contiguous portion of the
  // actual memory or a segment at the end and a segment at the
  // beginning.  This segment can grow or shrink at either end,
  // allowing insertions and removals at the ends without needing to
  // move any other entries.
  //
  // The start_position_ variable stores the index of the start of the
  // portion of the array that is in use, in the range [0, Capacity).
  // Care must be taken to keep this value in range when the start of
  // the used memory crosses the end of the storage.  The size_
  // variable stores the number of existing elements, which must be in
  // the range [0, Capacity].  (This could not be implemented using a
  // "one past the end" index instead of the size, because that index
  // is the same for size=0 and size=Capacity.)
  alignas(T) std::byte data_[sizeof(T) * Capacity];
  size_type start_position_{0};
  size_type size_{0};
};

template <typename T, size_t Capacity>
StaticDeque<T, Capacity>::StaticDeque(const size_type count) {
  resize(count);
}

template <typename T, size_t Capacity>
StaticDeque<T, Capacity>::StaticDeque(const size_type n, const T& value) {
  resize(n, value);
}

template <typename T, size_t Capacity>
template <typename InputIterator,
          Requires<static_deque_detail::is_input_iterator<InputIterator>>>
StaticDeque<T, Capacity>::StaticDeque(const InputIterator first,
                                      const InputIterator last) {
  assign(first, last);
}

template <typename T, size_t Capacity>
StaticDeque<T, Capacity>::StaticDeque(const StaticDeque& other)
    : StaticDeque(other.begin(), other.end()) {}

template <typename T, size_t Capacity>
StaticDeque<T, Capacity>::StaticDeque(StaticDeque&& other)
    : StaticDeque(std::move_iterator(other.begin()),
                  std::move_iterator(other.end())) {}

template <typename T, size_t Capacity>
StaticDeque<T, Capacity>::StaticDeque(const std::initializer_list<T> init)
    : StaticDeque(init.begin(), init.end()) {}

template <typename T, size_t Capacity>
StaticDeque<T, Capacity>::~StaticDeque() {
  clear();
}

template <typename T, size_t Capacity>
auto StaticDeque<T, Capacity>::operator=(const StaticDeque& other)
    -> StaticDeque& {
  if (size() > other.size()) {
    resize(other.size());
  }
  size_t i = 0;
  for (; i < size(); ++i) {
    (*this)[i] = other[i];
  }
  for (; i < other.size(); ++i) {
    push_back(other[i]);
  }
  return *this;
}

template <typename T, size_t Capacity>
auto StaticDeque<T, Capacity>::operator=(StaticDeque&& other) -> StaticDeque& {
  if (size() > other.size()) {
    resize(other.size());
  }
  size_t i = 0;
  for (; i < size(); ++i) {
    (*this)[i] = std::move(other[i]);
  }
  for (; i < other.size(); ++i) {
    push_back(std::move(other[i]));
  }
  return *this;
}

template <typename T, size_t Capacity>
auto StaticDeque<T, Capacity>::operator=(const std::initializer_list<T> init)
    -> StaticDeque& {
  assign(init);
  return *this;
}

template <typename T, size_t Capacity>
template <typename InputIterator,
          Requires<static_deque_detail::is_input_iterator<InputIterator>>>
void StaticDeque<T, Capacity>::assign(InputIterator first,
                                      const InputIterator last) {
  // This could be slightly more efficient if this object is not
  // already empty by assigning to existing objects.
  clear();
  while (first != last) {
    push_back(*first++);
  }
}

template <typename T, size_t Capacity>
void StaticDeque<T, Capacity>::assign(const size_type count, const T& value) {
  ASSERT(count <= max_size(), count << " exceeds max size " << max_size());
  // This could be slightly more efficient if this object is not
  // already empty by assigning to existing objects.
  clear();
  resize(count, value);
}

template <typename T, size_t Capacity>
void StaticDeque<T, Capacity>::assign(const std::initializer_list<T> init) {
  assign(init.begin(), init.end());
}

template <typename T, size_t Capacity>
void StaticDeque<T, Capacity>::resize(const size_type count) {
  ASSERT(count <= max_size(), count << " exceeds max size " << max_size());
  while (size() > count) {
    pop_back();
  }
  while (size() < count) {
    emplace_back();
  }
}

template <typename T, size_t Capacity>
void StaticDeque<T, Capacity>::resize(const size_type count, const T& value) {
  ASSERT(count <= max_size(), count << " exceeds max size " << max_size());
  while (size() > count) {
    pop_back();
  }
  while (size() < count) {
    push_back(value);
  }
}

template <typename T, size_t Capacity>
template <typename... Args>
T& StaticDeque<T, Capacity>::emplace_front(Args&&... args) {
  ASSERT(size() < max_size(), "Container is full.");
  auto* const new_entry =
      new (entry_address(Capacity - 1)) T(std::forward<Args>(args)...);
  // This must be done last in case T's constructor throws an
  // exception.
  if (start_position_ == 0) {
    start_position_ = Capacity - 1;
  } else {
    --start_position_;
  }
  ++size_;
  return *new_entry;
}

template <typename T, size_t Capacity>
template <typename... Args>
T& StaticDeque<T, Capacity>::emplace_back(Args&&... args) {
  ASSERT(size() < max_size(), "Container is full.");
  auto* const new_entry =
      new (entry_address(size())) T(std::forward<Args>(args)...);
  // This must be done last in case T's constructor throws an
  // exception.
  ++size_;
  return *new_entry;
}

template <typename T, size_t Capacity>
template <typename... Args>
auto StaticDeque<T, Capacity>::emplace(const const_iterator position,
                                       Args&&... args) -> iterator {
  if (position == this->begin()) {
    emplace_front(std::forward<Args>(args)...);
    return this->begin();
  } else if (position == this->end()) {
    emplace_back(std::forward<Args>(args)...);
    return this->end() - 1;
  }

  const auto element_to_assign = position - this->begin();
  if (should_operate_on_back(position, position)) {
    emplace_back(std::move(this->back()));
    std::move_backward(this->begin() + element_to_assign, this->end() - 2,
                       this->end() - 1);
  } else {
    emplace_front(std::move(this->front()));
    std::move(this->begin() + 2, this->begin() + element_to_assign + 1,
              this->begin() + 1);
  }

  const auto new_element = this->begin() + element_to_assign;
  // emplace in the middle does not construct in place.  STL
  // containers also have this behavior.
  *new_element = T(std::forward<Args>(args)...);
  return new_element;
}

template <typename T, size_t Capacity>
auto StaticDeque<T, Capacity>::insert(const const_iterator position, const T& x)
    -> iterator {
  return emplace(position, x);
}

template <typename T, size_t Capacity>
auto StaticDeque<T, Capacity>::insert(const const_iterator position, T&& x)
    -> iterator {
  return emplace(position, std::move(x));
}

template <typename T, size_t Capacity>
auto StaticDeque<T, Capacity>::insert(const const_iterator position,
                                      const size_type n, const T& x)
    -> iterator {
  // In theory we could insert these in place and save a bunch of
  // moves, but keeping track of when elements must be assigned
  // vs. constructed would be a pain, and this can't be that common an
  // operation.
  if (should_operate_on_back(position, position)) {
    // We are about to invalidate position.
    const auto insert_index = position - this->begin();
    const auto old_size = static_cast<difference_type>(size());
    for (size_type i = 0; i < n; ++i) {
      emplace_back(x);
    }
    const auto new_position = this->begin() + insert_index;
    const auto old_end = this->begin() + old_size;
    std::rotate(new_position, old_end, this->end());
    return new_position;
  } else {
    // We are about to invalidate position.
    const auto insert_index = position - this->begin();
    const auto insert_back_index = this->end() - position;
    const auto old_size = static_cast<difference_type>(size());
    for (size_type i = 0; i < n; ++i) {
      emplace_front(x);
    }
    const auto new_position = this->begin() + insert_index;
    const auto new_position_end = this->end() - insert_back_index;
    const auto old_begin = this->end() - old_size;
    std::rotate(this->begin(), old_begin, new_position_end);
    return new_position;
  }
}

template <typename T, size_t Capacity>
template <typename InputIterator,
          Requires<static_deque_detail::is_input_iterator<InputIterator>>>
auto StaticDeque<T, Capacity>::insert(const const_iterator position,
                                      InputIterator first,
                                      const InputIterator last) -> iterator {
  // The iterators could be single-pass, so we can't clear space and
  // then insert the new elements.  We also can't do repeated single
  // insertions because that's too inefficient.  Instead, we have to
  // create the new elements at the end and then rotate them into
  // place.
  if (should_operate_on_back(position, position)) {
    // We are about to invalidate position.
    const auto insert_index = position - this->begin();
    const auto old_size = static_cast<difference_type>(size());
    while (first != last) {
      emplace_back(*first++);
    }
    const auto new_position = this->begin() + insert_index;
    const auto old_end = this->begin() + old_size;
    std::rotate(new_position, old_end, this->end());
    return new_position;
  } else {
    // We are about to invalidate position.
    const auto insert_index = position - this->begin();
    const auto insert_back_index = this->end() - position;
    const auto old_size = static_cast<difference_type>(size());
    while (first != last) {
      emplace_front(*first++);
    }
    const auto new_position = this->begin() + insert_index;
    const auto new_position_end = this->end() - insert_back_index;
    const auto old_begin = this->end() - old_size;
    std::reverse(this->begin(), old_begin);
    std::rotate(this->begin(), old_begin, new_position_end);
    return new_position;
  }
}

template <typename T, size_t Capacity>
auto StaticDeque<T, Capacity>::insert(const const_iterator position,
                                      const std::initializer_list<T> init)
    -> iterator {
  return insert(position, init.begin(), init.end());
}

template <typename T, size_t Capacity>
void StaticDeque<T, Capacity>::pop_front() {
  ASSERT(not this->empty(), "Container is empty.");
  if (start_position_ == Capacity - 1) {
    start_position_ = 0;
  } else {
    ++start_position_;
  }
  --size_;
  // Calling the destructor is sufficient to remove the element.
  // There is no special "placement delete".
  entry_address(Capacity - 1)->~T();
}

template <typename T, size_t Capacity>
void StaticDeque<T, Capacity>::pop_back() {
  ASSERT(not this->empty(), "Container is empty.");
  --size_;
  // Calling the destructor is sufficient to remove the element.
  // There is no special "placement delete".
  entry_address(size())->~T();
}

template <typename T, size_t Capacity>
auto StaticDeque<T, Capacity>::erase(const const_iterator position)
    -> iterator {
  return erase(position, position + 1);
}

template <typename T, size_t Capacity>
auto StaticDeque<T, Capacity>::erase(const const_iterator first,
                                     const const_iterator last) -> iterator {
  const auto result_offset = first - this->begin();
  auto num_to_remove = last - first;
  if (num_to_remove == 0) {
    return last - this->begin() + this->begin();
  }
  if (should_operate_on_back(first, last)) {
    std::move(last, this->cend(), first - this->begin() + this->begin());
    while (num_to_remove-- != 0) {
      pop_back();
    }
  } else {
    std::move_backward(this->cbegin(), first,
                       last - this->begin() + this->begin());
    while (num_to_remove-- != 0) {
      pop_front();
    }
  }
  return this->begin() + result_offset;
}

template <typename T, size_t Capacity>
void StaticDeque<T, Capacity>::swap(StaticDeque& other) {
  using std::swap;
  swap(*this, other);
}

template <typename T, size_t Capacity>
void StaticDeque<T, Capacity>::clear() {
  while (not this->empty()) {
    pop_back();
  }
}

template <typename T, size_t Capacity>
void StaticDeque<T, Capacity>::pup(PUP::er& p) {
  // If performance when serializing simple types becomes an issue, we
  // can just send the array of bytes and the other member variables,
  // which will probably perform better (unless, perhaps, the
  // container is nearly empty).  That can't be done for types that
  // are not trivially copyable, which would still need to use
  // something like the current implementation.
  size_type sz = size();
  p | sz;
  resize(sz);
  for (auto& entry : *this) {
    p | entry;
  }
}

template <typename T, size_t Capacity>
void swap(StaticDeque<T, Capacity>& x, StaticDeque<T, Capacity>& y) {
  const auto initial_size = x.size();
  if (initial_size > y.size()) {
    return swap(y, x);
  }
  for (size_t i = 0; i < initial_size; ++i) {
    using std::swap;
    swap(x[i], y[i]);
  }
  const auto to_move =
      y.begin() +
      static_cast<typename StaticDeque<T, Capacity>::difference_type>(
          initial_size);
  x.insert(x.end(), std::move_iterator(to_move), std::move_iterator(y.end()));
  y.erase(to_move, y.end());
}

template <typename T, size_t Capacity>
std::ostream& operator<<(std::ostream& s,
                         const StaticDeque<T, Capacity>& deque) {
  sequence_print_helper(s, deque.begin(), deque.end());
  return s;
}
