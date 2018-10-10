// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <cstddef>
#include <iterator>
#include <pup.h>
#include <pup_stl.h>
#include <type_traits>

#include "ErrorHandling/Assert.hpp"
#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep
#include "Utilities/Algorithm.hpp"
#include "Utilities/BoostHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/PrintHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
class FixedHashMapIterator;
/// \endcond

namespace FixedHashMap_detail {
template <typename T, typename = cpp17::void_t<>>
struct has_is_perfect_member : std::false_type {};

template <typename T>
struct has_is_perfect_member<T,
                             cpp17::void_t<decltype(T::template is_perfect<0>)>>
    : std::true_type {};

template <typename T>
constexpr bool has_is_perfect_member_v = has_is_perfect_member<T>::value;

template <class T, size_t MaxSize,
          bool HasIsPerfect = has_is_perfect_member_v<T>>
constexpr bool is_perfect = T::template is_perfect<MaxSize>;

template <class T, size_t MaxSize>
constexpr bool is_perfect<T, MaxSize, false> = false;
}  // namespace FixedHashMap_detail

/*!
 * \ingroup DataStructuresGroup
 * \brief A hash table with a compile-time specified maximum size and ability to
 * efficiently handle perfect hashes.
 *
 * There are a few requirements on the types passed to `FixedHashMap`. These
 * are:
 * - `Key` is CopyConstructible for copy and move semantics, as well as
 *   serialization
 * - `ValueType` is MoveConstructible
 * - If `Hash` is a perfect hash for some values of `MaxSize` then in order to
 *   use the perfect hashing improvements `Hash` must have a member variable
 *   template `is_perfect` that takes as its only template parameter a `size_t`
 *   equal to `MaxSize`. A perfect hash must map each `Key` to `[0, MaxSize)`
 *   uniquely.
 * - Keys are distinct.
 *
 * The interface is similar to std::unordered_map, so see the documentation for
 * that for details.
 *
 * #### Implementation Details
 *
 * - Uses linear probing for non-perfect hashes.
 * - The hash `Hash` is assumed to be a perfect hash for `MaxSize` if
 *   `Hash::template is_perfect<MaxSize>` is true.
 */
template <size_t MaxSize, class Key, class ValueType,
          class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
class FixedHashMap {
  static constexpr bool hash_is_perfect =
      FixedHashMap_detail::is_perfect<Hash, MaxSize>;

 public:
  using key_type = Key;
  using mapped_type = ValueType;
  using value_type = std::pair<const key_type, mapped_type>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using hasher = Hash;
  using key_equal = KeyEqual;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator =
      FixedHashMapIterator<MaxSize, key_type, value_type, hasher, key_equal>;
  using const_iterator =
      FixedHashMapIterator<MaxSize, key_type, const value_type, hasher,
                           key_equal>;

  FixedHashMap() = default;
  FixedHashMap(std::initializer_list<value_type> init) noexcept;
  FixedHashMap(const FixedHashMap&) = default;
  FixedHashMap& operator=(const FixedHashMap& other) noexcept(
      noexcept(std::is_nothrow_copy_constructible<value_type>::value));
  FixedHashMap(FixedHashMap&&) = default;
  FixedHashMap& operator=(FixedHashMap&& other) noexcept;
  ~FixedHashMap() = default;

  iterator begin() noexcept { return unconst(cbegin()); }
  const_iterator begin() const noexcept {
    return const_iterator(data_.begin(), data_.end());
  }
  const_iterator cbegin() const noexcept { return begin(); }

  iterator end() noexcept { return unconst(cend()); }
  const_iterator end() const noexcept {
    return const_iterator(data_.end(), data_.end());
  }
  const_iterator cend() const noexcept { return end(); }

  bool empty() const noexcept { return size_ == 0; }
  size_t size() const noexcept { return size_; }

  void clear() noexcept;

  // @{
  /// Inserts the element if it does not exists.
  std::pair<iterator, bool> insert(const value_type& value) noexcept {
    return insert(value_type(value));
  }
  std::pair<iterator, bool> insert(value_type&& value) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return insert_or_assign_impl<false>(const_cast<key_type&&>(value.first),
                                        std::move(value.second));
  }
  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) noexcept {
    return insert(value_type(std::forward<Args>(args)...));
  }
  // @}
  // @{
  /// Inserts the element if it does not exists, otherwise assigns to it the new
  /// value.
  template <class M>
  std::pair<iterator, bool> insert_or_assign(const key_type& key,
                                             M&& obj) noexcept {
    return insert_or_assign(key_type{key}, std::forward<M>(obj));
  }
  template <class M>
  std::pair<iterator, bool> insert_or_assign(key_type&& key, M&& obj) noexcept {
    return insert_or_assign_impl<true>(std::move(key), std::forward<M>(obj));
  }
  // @}

  iterator erase(const const_iterator& pos) noexcept;
  size_t erase(const key_type& key) noexcept;

  mapped_type& at(const key_type& key);
  const mapped_type& at(const key_type& key) const;
  mapped_type& operator[](const key_type& key) noexcept;

  size_t count(const key_type& key) const noexcept;
  iterator find(const key_type& key) noexcept {
    return unconst(cpp17::as_const(*this).find(key));
  }
  const_iterator find(const key_type& key) const noexcept {
    auto it = get_data_entry(key);
    return it != data_.end() and is_set(*it) ? const_iterator(&*it, data_.end())
                                             : end();
  }

  /// Check if `key` is in the map
  bool contains(const key_type& key) const noexcept {
    return find(key) != end();
  }

  /// Get key equal function object
  key_equal key_eq() const noexcept { return key_equal{}; }
  /// Get hash function object
  hasher hash_function() const noexcept { return hasher{}; }

  void pup(PUP::er& p) noexcept {  // NOLINT(google-runtime-references)
    p | data_;
    p | size_;
  }

 private:
  template <size_t FMaxSize, class FKey, class FValueType, class FHash,
            class FKeyEqual>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(
      const FixedHashMap<FMaxSize, FKey, FValueType, FHash, FKeyEqual>& a,
      const FixedHashMap<FMaxSize, FKey, FValueType, FHash, FKeyEqual>&
          b) noexcept;

  template <bool Assign, class M>
  std::pair<iterator, bool> insert_or_assign_impl(key_type&& key,
                                                  M&& obj) noexcept;

  using storage_type = std::array<boost::optional<value_type>, MaxSize>;

  SPECTRE_ALWAYS_INLINE size_type hash(const key_type& key) const noexcept {
    return hash_is_perfect ? Hash{}(key) : Hash{}(key) % MaxSize;
  }

  template <bool IsInserting>
  typename storage_type::iterator get_data_entry(const Key& key) noexcept;

  typename storage_type::const_iterator get_data_entry(const Key& key) const
      noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<FixedHashMap&>(*this).get_data_entry<false>(key);
  }

  iterator unconst(const const_iterator& it) noexcept {
    return iterator(&it.get_optional() - data_.begin() + data_.begin(),
                    data_.end());
  }

  SPECTRE_ALWAYS_INLINE static bool is_set(
      const boost::optional<value_type>& opt) noexcept {
    return static_cast<bool>(opt);
  }

  storage_type data_{};
  size_t size_ = 0;
};

/// \cond HIDDEN_SYMBOLS
template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
class FixedHashMapIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = std::remove_const_t<ValueType>;
  using difference_type = ptrdiff_t;
  using pointer = ValueType*;
  using reference = ValueType&;

  FixedHashMapIterator() = default;

  /// Implicit conversion from mutable to const iterator.
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator FixedHashMapIterator<MaxSize, Key, const ValueType, Hash, KeyEqual>()
      const noexcept {
    return {entry_, end_};
  }

  reference operator*() const noexcept {
    ASSERT(entry_ != nullptr, "Invalid FixedHashMapIterator");
    ASSERT(is_set(*entry_),
           "FixedHashMapIterator points to an invalid value in the map.");
    // deference pointer, then dereference boost::optional
    return **entry_;
  }
  pointer operator->() const noexcept {
    ASSERT(entry_ != nullptr, "Invalid FixedHashMapIterator");
    ASSERT(is_set(*entry_),
           "FixedHashMapIterator points to an invalid value in the map.");
    return entry_->get_ptr();
  }

  FixedHashMapIterator& operator++() noexcept;
  // clang-tidy wants this to return a const object.  Returning const
  // objects is very strange, and as of June 2018 clang-tidy's
  // explanation for the lint is a dead link.
  FixedHashMapIterator operator++(int) noexcept;  // NOLINT(cert-dcl21-cpp)

 private:
  friend class FixedHashMap<MaxSize, Key, typename value_type::second_type,
                            Hash, KeyEqual>;
  friend class FixedHashMapIterator<MaxSize, Key, value_type, Hash, KeyEqual>;

  friend bool operator==(const FixedHashMapIterator& a,
                         const FixedHashMapIterator& b) noexcept {
    return a.entry_ == b.entry_;
  }
  friend bool operator<(const FixedHashMapIterator& a,
                        const FixedHashMapIterator& b) noexcept {
    return a.entry_ < b.entry_;
  }

  // The rest of the comparison operators don't need access to the
  // class internals, but they need to be instantiated for each
  // instantiation of the class.
  friend bool operator!=(const FixedHashMapIterator& a,
                         const FixedHashMapIterator& b) noexcept {
    return not(a == b);
  }
  friend bool operator>(const FixedHashMapIterator& a,
                        const FixedHashMapIterator& b) noexcept {
    return b < a;
  }
  friend bool operator<=(const FixedHashMapIterator& a,
                         const FixedHashMapIterator& b) noexcept {
    return not(b < a);
  }
  friend bool operator>=(const FixedHashMapIterator& a,
                         const FixedHashMapIterator& b) noexcept {
    return not(a < b);
  }

  using map_optional_type = boost::optional<std::remove_const_t<ValueType>>;
  using optional_type =
      std::conditional_t<std::is_const<ValueType>::value,
                         const map_optional_type, map_optional_type>;

  SPECTRE_ALWAYS_INLINE static bool is_set(const optional_type& opt) noexcept {
    return static_cast<bool>(opt);
  }

  FixedHashMapIterator(optional_type* const entry,
                       optional_type* const end) noexcept
      : entry_(entry), end_(end) {
    if (entry_ != end_ and not*entry_) {
      operator++();
    }
  }

  optional_type& get_optional() const { return *entry_; }

  optional_type* entry_{nullptr};
  optional_type* end_{nullptr};
};

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
FixedHashMapIterator<MaxSize, Key, ValueType, Hash, KeyEqual>&
FixedHashMapIterator<MaxSize, Key, ValueType, Hash, KeyEqual>::
operator++() noexcept {
  ASSERT(entry_ != end_,
         "Tried to increment an end iterator, which is undefined behavior.");
  // Move to next element, which might not be the next element in the std::array
  do {
    ++entry_;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  } while (entry_ < end_ and not is_set(*entry_));
  return *this;
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
// NOLINTNEXTLINE(cert-dcl21-cpp) see declaration
FixedHashMapIterator<MaxSize, Key, ValueType, Hash, KeyEqual>
FixedHashMapIterator<MaxSize, Key, ValueType, Hash, KeyEqual>::operator++(
    int) noexcept {
  const auto ret = *this;
  operator++();
  return ret;
}
/// \endcond

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>&
FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>::operator=(
    const FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>&
        other) noexcept(noexcept(std::
                                     is_nothrow_copy_constructible<
                                         value_type>::value)) {
  if (this == &other) {
    return *this;
  }
  clear();
  size_ = other.size_;
  for (size_t i = 0; i < data_.size(); ++i) {
    const auto& other_optional = gsl::at(other.data_, i);
    if (is_set(other_optional)) {
      // The boost::optionals cannot be assigned to because they
      // contain the map keys, which are const, so we have to replace
      // the contents instead, since that counts as a destroy +
      // initialize.
      gsl::at(data_, i).emplace(*other_optional);
    }
  }
  return *this;
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>&
FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>::operator=(
    FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  clear();
  size_ = other.size_;
  for (size_t i = 0; i < data_.size(); ++i) {
    auto& other_optional = gsl::at(other.data_, i);
    if (is_set(other_optional)) {
      // The boost::optionals cannot be assigned to because they
      // contain the map keys, which are const, so we have to replace
      // the contents instead, since that counts as a destroy +
      // initialize.
      gsl::at(data_, i).emplace(std::move(*other_optional));
    }
  }
  return *this;
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>::FixedHashMap(
    std::initializer_list<value_type> init) noexcept {
  // size_ is set by calling insert
  for (const auto& entry : init) {
    insert(entry);
  }
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
void FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>::clear() noexcept {
  for (auto& entry : data_) {
    entry = boost::none;
  }
  size_ = 0;
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
template <bool Assign, class M>
auto FixedHashMap<MaxSize, Key, ValueType, Hash,
                  KeyEqual>::insert_or_assign_impl(key_type&& key,
                                                   M&& obj) noexcept
    -> std::pair<iterator, bool> {
  auto data_it = get_data_entry<true>(key);
  if (UNLIKELY(data_it == data_.end())) {
    ERROR("Unable to insert element into FixedHashMap of maximum size "
          << MaxSize << " because it is full. If the current size (" << size_
          << ") is not equal to the maximum size then please file an issue "
             "with the code you used to produce this error. "
          << (hash_is_perfect ? " If a perfect hash is used and it hashes to "
                                "past the last element stored, then this "
                                "error may also be triggered."
                              : ""));
  }
  // data_it points to either the existing element or a new bucket to place the
  // element.
  const auto is_new_entry = not is_set(*data_it);
  if (is_new_entry or Assign) {
    if (is_new_entry) {
      ++size_;
    }
    data_it->emplace(std::move(key), std::forward<M>(obj));
  }
  return std::make_pair(iterator{data_it, data_.end()}, is_new_entry);
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
auto FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>::erase(
    const const_iterator& pos) noexcept -> iterator {
  auto next_it = &(unconst(pos).get_optional());
  --size_;
  *next_it = boost::none;
  // pos now points to an unset entry, advance to next valid one
  do {
    ++next_it;
    if (next_it == data_.end()) {
      next_it = data_.begin();
    }
    if (is_set(*next_it)) {
      return {next_it, data_.end()};
    }
  } while (next_it != &pos.get_optional());
  return end();
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
size_t FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>::erase(
    const key_type& key) noexcept {
  auto it = get_data_entry<false>(key);
  if (it == data_.end() or not is_set(*it)) {
    return 0;
  }
  *it = boost::none;
  --size_;
  return 1;
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
auto FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>::at(
    const key_type& key) -> mapped_type& {
  auto it = get_data_entry<false>(key);
  if (it == data_.end() or not is_set(*it)) {
    throw std::out_of_range(get_output(key) + " not in FixedHashMap");
  }
  return (**it).second;
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
auto FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>::at(
    const key_type& key) const -> const mapped_type& {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<FixedHashMap&>(*this).at(key);
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
auto FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>::operator[](
    const key_type& key) noexcept -> mapped_type& {
  auto it = get_data_entry<true>(key);
  if (it == data_.end()) {
    ERROR("Unable to insert element into FixedHashMap of maximum size "
          << MaxSize << " because it is full. If the current size (" << size_
          << ") is not equal to the maximum size then please file an issue "
             "with the code you used to produce this error. "
          << (hash_is_perfect ? " If a perfect hash is used and it hashes to "
                                "past the last element stored, then this "
                                "error may also be triggered."
                              : ""));
  }
  if (not is_set(*it)) {
    ++size_;
    it->emplace(key, mapped_type{});
  }
  return (**it).second;
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
size_t FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>::count(
    const key_type& key) const noexcept {
  const auto it = get_data_entry(key);
  return it == data_.end() or not is_set(*it) ? 0 : 1;
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
template <bool IsInserting>
auto FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>::get_data_entry(
    const Key& key) noexcept -> typename storage_type::iterator {
  const auto hashed_key = hash(key);
  auto it = data_.begin() + hashed_key;
  if (not hash_is_perfect) {
    // First search for an existing key.
    while (not is_set(*it) or (**it).first != key) {
      if (++it == data_.end()) {
        it = data_.begin();
      }
      if (UNLIKELY(it == data_.begin() + hashed_key)) {
        // not found, return end iterator to storage_type
        it = data_.end();
        break;
      }
    }
    // If we are inserting an element but couldn't find an existing key, then
    // find the first available bucket.
    if (IsInserting and it == data_.end()) {
      it = data_.begin() + hashed_key;
      while (is_set(*it)) {
        if (++it == data_.end()) {
          it = data_.begin();
        }
        if (UNLIKELY(it == data_.begin() + hashed_key)) {
          // could not find any open bucket
          return data_.end();
        }
      }
    }
  }
  return it;
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
bool operator==(
    const FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>& a,
    const FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>& b) noexcept {
  return a.size_ == b.size_ and a.data_ == b.data_;
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
bool operator!=(
    const FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>& a,
    const FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>& b) noexcept {
  return not(a == b);
}

template <size_t MaxSize, class Key, class ValueType, class Hash,
          class KeyEqual>
std::ostream& operator<<(
    std::ostream& os,
    const FixedHashMap<MaxSize, Key, ValueType, Hash, KeyEqual>& m) noexcept {
  unordered_print_helper(
      os, std::begin(m), std::end(m),
      [](std::ostream & out,
         typename FixedHashMap<MaxSize, Key, ValueType, Hash,
                               KeyEqual>::const_iterator it) noexcept {
        out << "[" << it->first << "," << it->second << "]";
      });
  return os;
}
