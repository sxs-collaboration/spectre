// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <utility>

#include "Domain/Direction.hpp"
#include "Domain/Side.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep
#include "Utilities/BoostHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrintHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \cond
template <typename ValueType>
class DirectionMapIterator;
/// \endcond

/// \ingroup DataStructuresGroup
/// \ingroup ComputationalDomainGroup
/// An optimized map with Direction keys.
///
/// Behaves like a `std::unordered_map<Direction<Dim>, T>` except that
/// all items are stored on the stack.  %DirectionMap should be faster
/// than unordered_map in most cases.
template <size_t Dim, typename T>
class DirectionMap {
 public:
  using key_type = Direction<Dim>;
  using mapped_type = T;
  using value_type = std::pair<const key_type, mapped_type>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = DirectionMapIterator<value_type>;
  using const_iterator = DirectionMapIterator<const value_type>;

  DirectionMap() = default;
  DirectionMap(const DirectionMap&) = default;
  DirectionMap(DirectionMap&&) = default;
  ~DirectionMap() = default;

  DirectionMap& operator=(const DirectionMap& other) noexcept;
  DirectionMap& operator=(DirectionMap&& other) noexcept;

  DirectionMap(std::initializer_list<value_type> init) noexcept;

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

  bool empty() const noexcept {
    return std::none_of(data_.begin(), data_.end(), is_set);
  }
  size_t size() const noexcept {
    return static_cast<size_t>(
        std::count_if(data_.begin(), data_.end(), is_set));
  }

  void clear() noexcept;
  std::pair<iterator, bool> insert(const value_type& value) noexcept {
    return insert(value_type(value));
  }
  std::pair<iterator, bool> insert(value_type&& value) noexcept;
  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) noexcept {
    return insert(value_type(std::forward<Args>(args)...));
  }
  iterator erase(const const_iterator& pos) noexcept;
  size_t erase(const Direction<Dim>& key) noexcept;

  T& at(const Direction<Dim>& key);
  const T& at(const Direction<Dim>& key) const;
  T& operator[](const Direction<Dim>& key) noexcept;

  size_t count(const Direction<Dim>& key) const noexcept {
    return data_entry(key) ? 1 : 0;
  }
  iterator find(const Direction<Dim>& key) noexcept {
    return unconst(cpp17::as_const(*this).find(key));
  }
  const_iterator find(const Direction<Dim>& key) const noexcept {
    return data_entry(key) ? const_iterator(&data_entry(key), data_.end())
                           : end();
  }

  void pup(PUP::er& p) noexcept {  // NOLINT(google-runtime-references)
    p | data_;
  }

 private:
  boost::optional<value_type>& data_entry(const Direction<Dim>& key) noexcept {
    return gsl::at(data_,
                   2 * key.dimension() + (key.side() == Side::Upper ? 1 : 0));
  }
  const boost::optional<value_type>& data_entry(const Direction<Dim>& key) const
      noexcept {
    return gsl::at(data_,
                   2 * key.dimension() + (key.side() == Side::Upper ? 1 : 0));
  }

  iterator unconst(const const_iterator& it) noexcept {
    return iterator(&it.get_optional() - data_.begin() + data_.begin(),
                    data_.end());
  }

  static bool is_set(const boost::optional<value_type>& opt) noexcept {
    return static_cast<bool>(opt);
  }

  template <size_t FDim, typename FT>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const DirectionMap<FDim, FT>& a,
                         const DirectionMap<FDim, FT>& b) noexcept;

  std::array<boost::optional<value_type>, 2 * Dim> data_;
};

/// \cond HIDDEN_SYMBOLS
template <typename ValueType>
class DirectionMapIterator {
 public:
  using difference_type = ptrdiff_t;
  using value_type = std::remove_const_t<ValueType>;
  using pointer = ValueType*;
  using reference = ValueType&;
  using iterator_category = std::forward_iterator_tag;

  DirectionMapIterator() = default;

  /// Implicit conversion from mutable to const iterator.
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator DirectionMapIterator<const ValueType>() const noexcept {
    return {entry_, end_};
  }

  reference operator*() const noexcept { return **entry_; }
  pointer operator->() const noexcept { return entry_->get_ptr(); }

  DirectionMapIterator& operator++() noexcept;
  // clang-tidy wants this to return a const object.  Returning const
  // objects is very strange, and as of June 2018 clang-tidy's
  // explanation for the lint is a dead link.
  DirectionMapIterator operator++(int) noexcept;  // NOLINT(cert-dcl21-cpp)

 private:
  friend class DirectionMap<value_type::first_type::volume_dim,
                            typename value_type::second_type>;
  friend class DirectionMapIterator<value_type>;

  friend bool operator==(const DirectionMapIterator& a,
                         const DirectionMapIterator& b) noexcept {
    return a.entry_ == b.entry_;
  }
  friend bool operator<(const DirectionMapIterator& a,
                        const DirectionMapIterator& b) noexcept {
    return a.entry_ < b.entry_;
  }

  // The rest of the comparison operators don't need access to the
  // class internals, but they need to be instantiated for each
  // instantiation of the class.
  friend bool operator!=(const DirectionMapIterator& a,
                         const DirectionMapIterator& b) noexcept {
    return not(a == b);
  }
  friend bool operator>(const DirectionMapIterator& a,
                        const DirectionMapIterator& b) noexcept {
    return b < a;
  }
  friend bool operator<=(const DirectionMapIterator& a,
                         const DirectionMapIterator& b) noexcept {
    return not(b < a);
  }
  friend bool operator>=(const DirectionMapIterator& a,
                         const DirectionMapIterator& b) noexcept {
    return not(a < b);
  }

  using map_optional_type = boost::optional<std::remove_const_t<ValueType>>;
  using optional_type =
      tmpl::conditional_t<std::is_const<ValueType>::value,
                          const map_optional_type, map_optional_type>;

  DirectionMapIterator(optional_type* const entry,
                       optional_type* const end) noexcept
      : entry_(entry), end_(end) {
    if (entry_ != end_ and not *entry_) {
      operator++();
    }
  }

  optional_type& get_optional() const { return *entry_; }

  optional_type* entry_{nullptr};
  optional_type* end_{nullptr};
};

template <typename ValueType>
DirectionMapIterator<ValueType>& DirectionMapIterator<ValueType>::
operator++() noexcept {
  ASSERT(entry_ != end_, "Tried to increment an end iterator");
  do {
    ++entry_;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  } while (entry_ < end_ and not *entry_);
  return *this;
}

template <typename ValueType>
// NOLINTNEXTLINE(cert-dcl21-cpp) see declaration
DirectionMapIterator<ValueType> DirectionMapIterator<ValueType>::operator++(
    int) noexcept {
  const auto ret = *this;
  operator++();
  return ret;
}
/// \endcond

template <size_t Dim, typename T>
DirectionMap<Dim, T>& DirectionMap<Dim, T>::operator=(
    const DirectionMap<Dim, T>& other) noexcept {
  clear();
  for (size_t i = 0; i < data_.size(); ++i) {
    const auto& other_optional = gsl::at(other.data_, i);
    if (other_optional) {
      // The boost::optionals cannot be assigned to because they
      // contain the map keys, which are const, so we have to replace
      // the contents instead, since that counts as a destroy +
      // initialize.
      gsl::at(data_, i).emplace(*other_optional);
    }
  }
  return *this;
}

template <size_t Dim, typename T>
DirectionMap<Dim, T>& DirectionMap<Dim, T>::operator=(
    DirectionMap<Dim, T>&& other) noexcept {
  clear();
  for (size_t i = 0; i < data_.size(); ++i) {
    auto& other_optional = gsl::at(other.data_, i);
    if (other_optional) {
      // The boost::optionals cannot be assigned to because they
      // contain the map keys, which are const, so we have to replace
      // the contents instead, since that counts as a destroy +
      // initialize.
      gsl::at(data_, i).emplace(std::move(*other_optional));
    }
  }
  return *this;
}

template <size_t Dim, typename T>
DirectionMap<Dim, T>::DirectionMap(
    const std::initializer_list<value_type> init) noexcept {
  for (const auto& entry : init) {
    insert(entry);
  }
}

template <size_t Dim, typename T>
void DirectionMap<Dim, T>::clear() noexcept {
  for (auto& entry : data_) {
    entry = boost::none;
  }
}

template <size_t Dim, typename T>
std::pair<typename DirectionMap<Dim, T>::iterator, bool>
DirectionMap<Dim, T>::insert(value_type&& value) noexcept {
  const Direction<Dim> key = value.first;
  const bool is_new_entry = not data_entry(key);
  if (is_new_entry) {
    data_entry(key).emplace(key, std::move(value.second));
  }
  return std::make_pair(find(key), is_new_entry);
}

template <size_t Dim, typename T>
typename DirectionMap<Dim, T>::iterator DirectionMap<Dim, T>::erase(
    const const_iterator& pos) noexcept {
  unconst(pos).get_optional() = boost::none;
  // pos is now invalid.  The unconst operation will adjust it to the
  // next valid value.
  return unconst(pos);
}

template <size_t Dim, typename T>
size_t DirectionMap<Dim, T>::erase(const Direction<Dim>& key) noexcept {
  if (not data_entry(key)) {
    return 0;
  }
  data_entry(key) = boost::none;
  return 1;
}

template <size_t Dim, typename T>
T& DirectionMap<Dim, T>::at(const Direction<Dim>& key) {
  if (not data_entry(key)) {
    throw std::out_of_range(get_output(key) + " not in map");
  }
  return data_entry(key)->second;
}

template <size_t Dim, typename T>
const T& DirectionMap<Dim, T>::at(const Direction<Dim>& key) const {
  if (not data_entry(key)) {
    throw std::out_of_range(get_output(key) + " not in map");
  }
  return data_entry(key)->second;
}

template <size_t Dim, typename T>
T& DirectionMap<Dim, T>::operator[](const Direction<Dim>& key) noexcept {
  if (not data_entry(key)) {
    data_entry(key).emplace(key, T{});
  }
  return data_entry(key)->second;
}

template <size_t Dim, typename T>
bool operator==(const DirectionMap<Dim, T>& a,
                const DirectionMap<Dim, T>& b) noexcept {
  return a.data_ == b.data_;
}

template <size_t Dim, typename T>
bool operator!=(const DirectionMap<Dim, T>& a,
                const DirectionMap<Dim, T>& b) noexcept {
  return not(a == b);
}

template <size_t Dim, typename T>
inline std::ostream& operator<<(std::ostream& os,
                                const DirectionMap<Dim, T>& m) {
  unordered_print_helper(
      os, std::begin(m), std::end(m),
      [](std::ostream& out, typename DirectionMap<Dim, T>::const_iterator it) {
        out << "[" << it->first << "," << it->second << "]";
      });
  return os;
}
