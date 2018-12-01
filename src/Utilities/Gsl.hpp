
/// \file
/// Defines functions and classes from the GSL

#pragma once

#pragma GCC system_header

// The code in this file is adapted from Microsoft's GSL that can be found at
// https://github.com/Microsoft/GSL
// The original license and copyright are:
///////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Microsoft Corporation. All rights reserved.
//
// This code is licensed under the MIT License (MIT).
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////
// The code changes are because SpECTRE is not allowed to throw under any
// circumstances that cannot be guaranteed to be caught and so all throw's
// are replaced by hard errors (ERROR).

#include <algorithm>  // for lexicographical_compare
#include <array>      // for array
#include <cstddef>    // for ptrdiff_t, size_t, nullptr_t
#include <iterator>   // for reverse_iterator, distance, random_access_...
#include <limits>
#include <memory>  // for std::addressof
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "ErrorHandling/ExpectsAndEnsures.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"

#if defined(__clang__) || defined(__GNUC__)

/*!
 * \ingroup UtilitiesGroup
 * The if statement is expected to evaluate true most of the time
 */
#define LIKELY(x) __builtin_expect(!!(x), 1)

/*!
 * \ingroup UtilitiesGroup
 * The if statement is expected to evaluate false most of the time
 */
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#else
/*!
 * \ingroup UtilitiesGroup
 * The if statement is expected to evaluate true most of the time
 */
#define LIKELY(x) (x)

/*!
 * \ingroup UtilitiesGroup
 * The if statement is expected to evaluate false most of the time
 */
#define UNLIKELY(x) (x)
#endif

/*!
 * \ingroup UtilitiesGroup
 * \brief Implementations from the Guideline Support Library
 */
namespace gsl {

/*!
 * \ingroup UtilitiesGroup
 * \brief Cast `u` to a type `T` where the cast may result in narrowing
 */
template <class T, class U>
SPECTRE_ALWAYS_INLINE constexpr T narrow_cast(U&& u) noexcept {
  return static_cast<T>(std::forward<U>(u));
}

namespace gsl_detail {
template <class T, class U>
struct is_same_signedness
    : public std::integral_constant<bool, std::is_signed<T>::value ==
                                              std::is_signed<U>::value> {};
}  // namespace gsl_detail

/*!
 * \ingroup UtilitiesGroup
 * \brief A checked version of narrow_cast() that ERRORs if the cast changed
 * the value
 */
template <class T, class U>
SPECTRE_ALWAYS_INLINE T narrow(U u) {
  T t = narrow_cast<T>(u);
  if (static_cast<U>(t) != u) {
    ERROR("Failed to cast " << u << " of type " << pretty_type::get_name<U>()
                            << " to type " << pretty_type::get_name<T>());
  }
  if (not gsl_detail::is_same_signedness<T, U>::value and
      ((t < T{}) != (u < U{}))) {
    ERROR("Failed to cast " << u << " of type " << pretty_type::get_name<U>()
                            << " to type " << pretty_type::get_name<T>());
  }
  return t;
}

// @{
/*!
 * \ingroup UtilitiesGroup
 * \brief Retrieve a entry from a container, with checks in Debug mode that
 * the index being retrieved is valid.
 */
template <class T, std::size_t N, typename Size>
SPECTRE_ALWAYS_INLINE constexpr T& at(std::array<T, N>& arr, Size index) {
  Expects(index >= 0 and index < narrow_cast<Size>(N));
  return arr[static_cast<std::size_t>(index)];
}

template <class Cont, typename Size>
SPECTRE_ALWAYS_INLINE constexpr const typename Cont::value_type& at(
    const Cont& cont, Size index) {
  Expects(index >= 0 and index < narrow_cast<Size>(cont.size()));
  return cont[static_cast<typename Cont::size_type>(index)];
}

template <class T, typename Size>
SPECTRE_ALWAYS_INLINE constexpr const T& at(std::initializer_list<T> cont,
                                            Size index) {
  Expects(index >= 0 and index < narrow_cast<Size>(cont.size()));
  return *(cont.begin() + index);
}
// @}

namespace detail {
template <class T>
struct owner_impl {
  static_assert(std::is_same<T, const owner_impl<int*>&>::value,
                "You should not have an owning raw pointer, instead you should "
                "use std::unique_ptr or, sparingly, std::shared_ptr. If "
                "clang-tidy told you to use gsl::owner, then you should still "
                "use std::unique_ptr instead.");
  using type = T;
};
}  // namespace detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Mark a raw pointer as owning its data
 *
 * \warning You should never actually use `gsl::owner`. Instead you should use
 * `std::unique_ptr`, and if shared ownership is required, `std::shared_ptr`.
 */
template <class T, Requires<std::is_pointer<T>::value> = nullptr>
using owner = typename detail::owner_impl<T>::type;

/*!
 * \ingroup UtilitiesGroup
 * \brief Require a pointer to not be a `nullptr`
 *
 * Restricts a pointer or smart pointer to only hold non-null values.
 *
 * Has zero size overhead over `T`.
 *
 * If `T` is a pointer (i.e. `T == U*`) then
 * - allow construction from `U*`
 * - disallow construction from `nullptr_t`
 * - disallow default construction
 * - ensure construction from null `U*` fails
 * - allow implicit conversion to `U*`
 */
template <class T>
class not_null {
 public:
  static_assert(std::is_assignable<T&, std::nullptr_t>::value,
                "T cannot be assigned nullptr.");

  template <typename U, Requires<std::is_convertible<U, T>::value> = nullptr>
  constexpr not_null(U&& u) : ptr_(std::forward<U>(u)) {
    Expects(ptr_ != nullptr);
  }

  template <typename U, Requires<std::is_convertible<U, T>::value> = nullptr>
  constexpr not_null(const not_null<U>& other) : not_null(other.get()) {}

  not_null(const not_null& other) = default;
  not_null& operator=(const not_null& other) = default;

  constexpr T get() const {
    Ensures(ptr_ != nullptr);
    return ptr_;
  }

  constexpr operator T() const { return get(); }
  constexpr T operator->() const { return get(); }
  constexpr decltype(auto) operator*() const { return *get(); }

  // prevents compilation when someone attempts to assign a null pointer
  // constant
  not_null(std::nullptr_t) = delete;
  not_null& operator=(std::nullptr_t) = delete;

  // unwanted operators...pointers only point to single objects!
  not_null& operator++() = delete;
  not_null& operator--() = delete;
  not_null operator++(int) = delete;
  not_null operator--(int) = delete;
  not_null& operator+=(std::ptrdiff_t) = delete;
  not_null& operator-=(std::ptrdiff_t) = delete;
  void operator[](std::ptrdiff_t) const = delete;

 private:
  T ptr_;
};

template <class T>
std::ostream& operator<<(std::ostream& os, const not_null<T>& val) {
  os << val.get();
  return os;
}

template <class T, class U>
auto operator==(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() == rhs.get()) {
  return lhs.get() == rhs.get();
}

template <class T, class U>
auto operator!=(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() != rhs.get()) {
  return lhs.get() != rhs.get();
}

template <class T, class U>
auto operator<(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() < rhs.get()) {
  return lhs.get() < rhs.get();
}

template <class T, class U>
auto operator<=(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() <= rhs.get()) {
  return lhs.get() <= rhs.get();
}

template <class T, class U>
auto operator>(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() > rhs.get()) {
  return lhs.get() > rhs.get();
}

template <class T, class U>
auto operator>=(const not_null<T>& lhs, const not_null<U>& rhs)
    -> decltype(lhs.get() >= rhs.get()) {
  return lhs.get() >= rhs.get();
}

// more unwanted operators
template <class T, class U>
std::ptrdiff_t operator-(const not_null<T>&, const not_null<U>&) = delete;
template <class T>
not_null<T> operator-(const not_null<T>&, std::ptrdiff_t) = delete;
template <class T>
not_null<T> operator+(const not_null<T>&, std::ptrdiff_t) = delete;
template <class T>
not_null<T> operator+(std::ptrdiff_t, const not_null<T>&) = delete;

// GCC 7 does not like the signed unsigned missmatch (size_t ptrdiff_t)
// While there is a conversion from signed to unsigned, it happens at
// compile time, so the compiler wouldn't have to warn indiscriminently, but
// could check if the source value actually doesn't fit into the target type
// and only warn in those cases.
#if __GNUC__ > 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif  // __GNUC__ > 6

// [views.constants], constants
constexpr const std::ptrdiff_t dynamic_extent = -1;

template <class ElementType, std::ptrdiff_t Extent = dynamic_extent>
class span;

// implementation details
namespace detail {
template <class T>
struct is_span_oracle : std::false_type {};

template <class ElementType, std::ptrdiff_t Extent>
struct is_span_oracle<gsl::span<ElementType, Extent>> : std::true_type {};

template <class T>
struct is_span : public is_span_oracle<std::remove_cv_t<T>> {};

template <class T>
struct is_std_array_oracle : std::false_type {};

template <class ElementType, std::size_t Extent>
struct is_std_array_oracle<std::array<ElementType, Extent>> : std::true_type {};

template <class T>
struct is_std_array : public is_std_array_oracle<std::remove_cv_t<T>> {};

template <std::ptrdiff_t From, std::ptrdiff_t To>
struct is_allowed_extent_conversion
    : public std::integral_constant<bool, From == To ||
                                              From == gsl::dynamic_extent ||
                                              To == gsl::dynamic_extent> {};

template <class From, class To>
struct is_allowed_element_type_conversion
    : public std::integral_constant<
          bool, std::is_convertible<From (*)[], To (*)[]>::value> {};

template <class Span, bool IsConst>
class span_iterator {
  using element_type_ = typename Span::element_type;

 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = std::remove_cv_t<element_type_>;
  using difference_type = typename Span::index_type;

  using reference =
      std::conditional_t<IsConst, const element_type_, element_type_>&;
  using pointer = std::add_pointer_t<reference>;

  span_iterator() = default;

  constexpr span_iterator(const Span* span,
                          typename Span::index_type idx) noexcept
      : span_(span), index_(idx) {}

  friend span_iterator<Span, true>;
  template <bool B, Requires<!B && IsConst> = nullptr>
  constexpr span_iterator(const span_iterator<Span, B>& other) noexcept
      : span_iterator(other.span_, other.index_) {}

  constexpr reference operator*() const {
    Expects(index_ != span_->size());
    return *(span_->data() + index_);
  }

  constexpr pointer operator->() const {
    Expects(index_ != span_->size());
    return span_->data() + index_;
  }

  constexpr span_iterator& operator++() {
    Expects(0 <= index_ && index_ != span_->size());
    ++index_;
    return *this;
  }

  constexpr span_iterator operator++(int) {
    auto ret = *this;
    ++(*this);
    return ret;
  }

  constexpr span_iterator& operator--() {
    Expects(index_ != 0 && index_ <= span_->size());
    --index_;
    return *this;
  }

  constexpr span_iterator operator--(int) {
    auto ret = *this;
    --(*this);
    return ret;
  }

  constexpr span_iterator operator+(difference_type n) const {
    auto ret = *this;
    return ret += n;
  }

  friend constexpr span_iterator operator+(difference_type n,
                                           span_iterator const& rhs) {
    return rhs + n;
  }

  constexpr span_iterator& operator+=(difference_type n) {
    Expects((index_ + n) >= 0 && (index_ + n) <= span_->size());
    index_ += n;
    return *this;
  }

  constexpr span_iterator operator-(difference_type n) const {
    auto ret = *this;
    return ret -= n;
  }

  constexpr span_iterator& operator-=(difference_type n) { return *this += -n; }

  constexpr difference_type operator-(span_iterator rhs) const {
    Expects(span_ == rhs.span_);
    return index_ - rhs.index_;
  }

  constexpr reference operator[](difference_type n) const {
    return *(*this + n);
  }

  constexpr friend bool operator==(span_iterator lhs,
                                   span_iterator rhs) noexcept {
    return lhs.span_ == rhs.span_ && lhs.index_ == rhs.index_;
  }

  constexpr friend bool operator!=(span_iterator lhs,
                                   span_iterator rhs) noexcept {
    return !(lhs == rhs);
  }

  constexpr friend bool operator<(span_iterator lhs,
                                  span_iterator rhs) noexcept {
    return lhs.index_ < rhs.index_;
  }

  constexpr friend bool operator<=(span_iterator lhs,
                                   span_iterator rhs) noexcept {
    return !(rhs < lhs);
  }

  constexpr friend bool operator>(span_iterator lhs,
                                  span_iterator rhs) noexcept {
    return rhs < lhs;
  }

  constexpr friend bool operator>=(span_iterator lhs,
                                   span_iterator rhs) noexcept {
    return !(rhs > lhs);
  }

 protected:
  const Span* span_ = nullptr;
  std::ptrdiff_t index_ = 0;
};

template <std::ptrdiff_t Ext>
class extent_type {
 public:
  using index_type = std::ptrdiff_t;

  static_assert(Ext >= 0, "A fixed-size span must be >= 0 in size.");

  constexpr extent_type() noexcept {}

  template <index_type Other>
  constexpr extent_type(extent_type<Other> ext) {
    static_assert(
        Other == Ext || Other == dynamic_extent,
        "Mismatch between fixed-size extent and size of initializing data.");
    Expects(ext.size() == Ext);
  }

  constexpr extent_type(index_type size) { Expects(size == Ext); }

  constexpr index_type size() const noexcept { return Ext; }
};

template <>
class extent_type<dynamic_extent> {
 public:
  using index_type = std::ptrdiff_t;

  template <index_type Other>
  explicit constexpr extent_type(extent_type<Other> ext) : size_(ext.size()) {}

  explicit constexpr extent_type(index_type size) : size_(size) {
    Expects(size >= 0);
  }

  constexpr index_type size() const noexcept { return size_; }

 private:
  index_type size_;
};

template <class ElementType, std::ptrdiff_t Extent, std::ptrdiff_t Offset,
          std::ptrdiff_t Count>
struct calculate_subspan_type {
  using type =
      span<ElementType,
           Count != dynamic_extent
               ? Count
               : (Extent != dynamic_extent ? Extent - Offset : Extent)>;
};
}  // namespace detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Create a span/view on a range, which is cheap to copy (one pointer).
 */
template <class ElementType, std::ptrdiff_t Extent>
class span {
 public:
  // constants and types
  using element_type = ElementType;
  using value_type = std::remove_cv_t<ElementType>;
  using index_type = size_t;
  using pointer = element_type*;
  using reference = element_type&;

  using iterator = detail::span_iterator<span<ElementType, Extent>, false>;
  using const_iterator = detail::span_iterator<span<ElementType, Extent>, true>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  using size_type = index_type;

  static constexpr index_type extent{Extent};

  // [span.cons], span constructors, copy, assignment, and destructor
  template <bool Dependent = false,
            // "Dependent" is needed to make "Requires<Dependent ||
            // Extent <= 0>" SFINAE, since "Requires<Extent <= 0>" is
            // ill-formed when Extent is greater than 0.
            Requires<(Dependent || Extent <= 0)> = nullptr>
  constexpr span() noexcept : storage_(nullptr, detail::extent_type<0>()) {}

  constexpr span(pointer ptr, index_type count) : storage_(ptr, count) {}

  constexpr span(pointer firstElem, pointer lastElem)
      : storage_(firstElem, std::distance(firstElem, lastElem)) {}

  template <std::size_t N>
  constexpr span(element_type (&arr)[N]) noexcept
      : storage_(KnownNotNull{std::addressof(arr[0])},
                 detail::extent_type<N>()) {}

  template <std::size_t N, Requires<(N > 0)> = nullptr>
  constexpr span(std::array<std::remove_const_t<element_type>, N>& arr) noexcept
      : storage_(KnownNotNull{arr.data()}, detail::extent_type<N>()) {}

  constexpr span(std::array<std::remove_const_t<element_type>, 0>&) noexcept
      : storage_(static_cast<pointer>(nullptr), detail::extent_type<0>()) {}

  template <std::size_t N, Requires<(N > 0)> = nullptr>
  constexpr span(
      const std::array<std::remove_const_t<element_type>, N>& arr) noexcept
      : storage_(KnownNotNull{arr.data()}, detail::extent_type<N>()) {}

  constexpr span(
      const std::array<std::remove_const_t<element_type>, 0>&) noexcept
      : storage_(static_cast<pointer>(nullptr), detail::extent_type<0>()) {}

  // NB: the SFINAE here uses .data() as a incomplete/imperfect proxy for the
  // requirement on Container to be a contiguous sequence container.
  template <
      class Container,
      Requires<
          !detail::is_span<Container>::value &&
          !detail::is_std_array<Container>::value &&
          std::is_convertible<typename Container::pointer, pointer>::value &&
          std::is_convertible<
              typename Container::pointer,
              decltype(std::declval<Container>().data())>::value> = nullptr>
  constexpr span(Container& cont)
      : span(cont.data(), narrow<index_type>(cont.size())) {}

  template <
      class Container,
      Requires<
          std::is_const<element_type>::value &&
          !detail::is_span<Container>::value &&
          std::is_convertible<typename Container::pointer, pointer>::value &&
          std::is_convertible<
              typename Container::pointer,
              decltype(std::declval<Container>().data())>::value> = nullptr>
  constexpr span(const Container& cont)
      : span(cont.data(), narrow<index_type>(cont.size())) {}

  constexpr span(const span& other) noexcept = default;

  template <class OtherElementType, std::ptrdiff_t OtherExtent,
            Requires<detail::is_allowed_extent_conversion<OtherExtent,
                                                          Extent>::value &&
                     detail::is_allowed_element_type_conversion<
                         OtherElementType, element_type>::value> = nullptr>
  constexpr span(const span<OtherElementType, OtherExtent>& other)
      : storage_(other.data(), detail::extent_type<OtherExtent>(other.size())) {
  }

  ~span() noexcept = default;
  constexpr span& operator=(const span& other) noexcept = default;

  // [span.sub], span subviews
  template <std::ptrdiff_t Count>
  constexpr span<element_type, Count> first() const {
    Expects(Count >= 0 && Count <= size());
    return {data(), Count};
  }

  template <std::ptrdiff_t Count>
  constexpr span<element_type, Count> last() const {
    Expects(Count >= 0 && size() - Count >= 0);
    return {data() + (size() - Count), Count};
  }

  template <std::ptrdiff_t Offset, std::ptrdiff_t Count = dynamic_extent>
  constexpr auto subspan() const ->
      typename detail::calculate_subspan_type<ElementType, Extent, Offset,
                                              Count>::type {
    Expects(
        (Offset >= 0 && size() - Offset >= 0) &&
        (Count == dynamic_extent || (Count >= 0 && Offset + Count <= size())));

    return {data() + Offset, Count == dynamic_extent ? size() - Offset : Count};
  }

  constexpr span<element_type, dynamic_extent> first(index_type count) const {
    Expects(count >= 0 && count <= size());
    return {data(), count};
  }

  constexpr span<element_type, dynamic_extent> last(index_type count) const {
    return make_subspan(size() - count, dynamic_extent,
                        subspan_selector<Extent>{});
  }

  constexpr span<element_type, dynamic_extent> subspan(
      index_type offset, index_type count = dynamic_extent) const {
    return make_subspan(offset, count, subspan_selector<Extent>{});
  }

  // [span.obs], span observers
  constexpr index_type size() const noexcept { return storage_.size(); }
  constexpr index_type size_bytes() const noexcept {
    return size() * narrow_cast<index_type>(sizeof(element_type));
  }
  constexpr bool empty() const noexcept { return size() == 0; }

  // [span.elem], span element access
  constexpr reference operator[](index_type idx) const {
    Expects(CheckRange(idx, storage_.size()));
    return data()[idx];
  }

  constexpr reference at(index_type idx) const { return this->operator[](idx); }
  constexpr reference operator()(index_type idx) const {
    return this->operator[](idx);
  }
  constexpr pointer data() const noexcept { return storage_.data(); }

  // [span.iter], span iterator support
  constexpr iterator begin() const noexcept { return {this, 0}; }
  constexpr iterator end() const noexcept { return {this, size()}; }

  constexpr const_iterator cbegin() const noexcept { return {this, 0}; }
  constexpr const_iterator cend() const noexcept { return {this, size()}; }

  constexpr reverse_iterator rbegin() const noexcept {
    return reverse_iterator{end()};
  }
  constexpr reverse_iterator rend() const noexcept {
    return reverse_iterator{begin()};
  }

  constexpr const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator{cend()};
  }
  constexpr const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator{cbegin()};
  }

 private:
  static bool CheckRange(index_type idx, index_type size) {
    // Optimization:
    //
    // idx >= 0 && idx < size
    // =>
    // static_cast<size_t>(idx) < static_cast<size_t>(size)
    //
    // because size >=0 by span construction, and negative idx will
    // wrap around to a value always greater than size when casted.

    // check if we have enough space to wrap around
    if (sizeof(index_type) <= sizeof(size_t)) {
      return narrow_cast<size_t>(idx) < narrow_cast<size_t>(size);
    } else {
      return idx >= 0 && idx < size;
    }
  }

  // Needed to remove unnecessary null check in subspans
  struct KnownNotNull {
    pointer p;
  };

  // this implementation detail class lets us take advantage of the
  // empty base class optimization to pay for only storage of a single
  // pointer in the case of fixed-size spans
  template <class ExtentType>
  class storage_type : public ExtentType {
   public:
    // KnownNotNull parameter is needed to remove unnecessary null check
    // in subspans and constructors from arrays
    template <class OtherExtentType>
    constexpr storage_type(KnownNotNull data, OtherExtentType ext)
        : ExtentType(ext), data_(data.p) {
      Expects(ExtentType::size() >= 0);
    }

    template <class OtherExtentType>
    constexpr storage_type(pointer data, OtherExtentType ext)
        : ExtentType(ext), data_(data) {
      Expects(ExtentType::size() >= 0);
      Expects(data || ExtentType::size() == 0);
    }

    constexpr pointer data() const noexcept { return data_; }

   private:
    pointer data_;
  };

  storage_type<detail::extent_type<Extent>> storage_;

  // The rest is needed to remove unnecessary null check
  // in subspans and constructors from arrays
  constexpr span(KnownNotNull ptr, index_type count) : storage_(ptr, count) {}

  template <std::ptrdiff_t CallerExtent>
  class subspan_selector {};

  template <std::ptrdiff_t CallerExtent>
  span<element_type, dynamic_extent> make_subspan(
      index_type offset, index_type count,
      subspan_selector<CallerExtent>) const {
    const span<element_type, dynamic_extent> tmp(*this);
    return tmp.subspan(offset, count);
  }

  span<element_type, dynamic_extent> make_subspan(
      index_type offset, index_type count,
      subspan_selector<dynamic_extent>) const {
    Expects(offset >= 0 && size() - offset >= 0);

    if (count == dynamic_extent) {
      return {KnownNotNull{data() + offset}, size() - offset};
    }

    Expects(count >= 0 && size() - offset >= count);
    return {KnownNotNull{data() + offset}, count};
  }
};

// [span.comparison], span comparison operators
template <class ElementType, std::ptrdiff_t FirstExtent,
          std::ptrdiff_t SecondExtent>
constexpr bool operator==(span<ElementType, FirstExtent> l,
                          span<ElementType, SecondExtent> r) {
  return std::equal(l.begin(), l.end(), r.begin(), r.end());
}

template <class ElementType, std::ptrdiff_t Extent>
constexpr bool operator!=(span<ElementType, Extent> l,
                          span<ElementType, Extent> r) {
  return !(l == r);
}

template <class ElementType, std::ptrdiff_t Extent>
constexpr bool operator<(span<ElementType, Extent> l,
                         span<ElementType, Extent> r) {
  return std::lexicographical_compare(l.begin(), l.end(), r.begin(), r.end());
}

template <class ElementType, std::ptrdiff_t Extent>
constexpr bool operator<=(span<ElementType, Extent> l,
                          span<ElementType, Extent> r) {
  return !(l > r);
}

template <class ElementType, std::ptrdiff_t Extent>
constexpr bool operator>(span<ElementType, Extent> l,
                         span<ElementType, Extent> r) {
  return r < l;
}

template <class ElementType, std::ptrdiff_t Extent>
constexpr bool operator>=(span<ElementType, Extent> l,
                          span<ElementType, Extent> r) {
  return !(l < r);
}

// @{
/// \ingroup UtilitiesGroup
/// Utility function for creating spans
template <class ElementType>
constexpr span<ElementType> make_span(
    ElementType* ptr, typename span<ElementType>::index_type count) {
  return span<ElementType>(ptr, count);
}

template <class ElementType>
constexpr span<ElementType> make_span(ElementType* firstElem,
                                      ElementType* lastElem) {
  return span<ElementType>(firstElem, lastElem);
}

template <class ElementType, std::size_t N>
constexpr span<ElementType, N> make_span(ElementType (&arr)[N]) noexcept {
  return span<ElementType, N>(arr);
}

template <class Container>
constexpr span<typename Container::value_type> make_span(Container& cont) {
  return span<typename Container::value_type>(cont);
}

template <class Container>
constexpr span<const typename Container::value_type> make_span(
    const Container& cont) {
  return span<const typename Container::value_type>(cont);
}

template <class Ptr>
constexpr span<typename Ptr::element_type> make_span(Ptr& cont,
                                                     std::ptrdiff_t count) {
  return span<typename Ptr::element_type>(cont, count);
}

template <class Ptr>
constexpr span<typename Ptr::element_type> make_span(Ptr& cont) {
  return span<typename Ptr::element_type>(cont);
}
// @}

// Specialization of gsl::at for span
template <class ElementType, std::ptrdiff_t Extent>
constexpr ElementType& at(span<ElementType, Extent> s,
                          typename span<ElementType, Extent>::index_type i) {
  // No bounds checking here because it is done in span::operator[] called below
  return s[i];
}

#if __GNUC__ > 6
#pragma GCC diagnostic pop
#endif  // __GNUC__ > 6
}  // namespace gsl

// The remainder of this file is
// Distributed under the MIT License.
// See LICENSE.txt for details.

/// Construct a not_null from a pointer.  Often this will be done as
/// an implicit conversion, but it may be necessary to perform the
/// conversion explicitly when type deduction is desired.
///
/// \note This is not a standard GSL function, and so is not in the
/// gsl namespace.
template <typename T>
gsl::not_null<T*> make_not_null(T* ptr) noexcept {
  return gsl::not_null<T*>(ptr);
}
