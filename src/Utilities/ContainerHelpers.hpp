// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <type_traits>

#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

/// \ingroup UtilitiesGroup
/// \brief Callable struct which retrieves the `t.size()` for operand `t`. This
/// will cause a compiler error if no such function exists.
struct GetContainerSize {
  template <typename T>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(const T& t) noexcept {
    return t.size();
  }
};

/// \ingroup UtilitiesGroup
/// \brief Callable struct for the subscript operator. Returns `t[i]`
struct GetContainerElement {
  template <typename T>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(T& t,
                                                  const size_t i) noexcept {
    return t[i];
  }
};

namespace ContainerHelpers_detail {
// implementation struct for get_element and get_size
template <bool IsFundamentalOrComplexOfFundamental>
struct GetImpls;

template <>
struct GetImpls<true> {
  template <typename T, typename SubscriptFunction>
  static SPECTRE_ALWAYS_INLINE decltype(auto) get_element(
      T& t, const size_t /*i*/, const SubscriptFunction /*at*/) noexcept {
    return t;
  }

  template <typename T, typename SizeFunction>
  static SPECTRE_ALWAYS_INLINE size_t
  get_size(const T& /*t*/, const SizeFunction /*size*/) noexcept {
    return 1;
  }
};

template <>
struct GetImpls<false> {
  template <typename T, typename SubscriptFunction>
  static SPECTRE_ALWAYS_INLINE decltype(auto) get_element(
      T& t, const size_t i, SubscriptFunction at) noexcept {
    return at(t, i);
  }

  template <typename T, typename SizeFunction>
  static SPECTRE_ALWAYS_INLINE decltype(auto) get_size(
      const T& t, SizeFunction size) noexcept {
    return size(t);
  }
};
}  // namespace ContainerHelpers_detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Returns the `i`th element if `T` has a subscript operator, otherwise
 * if `T` is fundamental or a `std::complex` of a fundamental type,  returns
 * `t`.
 *
 * \details This function also optionally takes the user-defined subscript
 * function `at`, which can be used to specify a custom indexing function. For
 * instance, for a type which is a `std::array` of a `std::array`s, the indexing
 * function could be the below callable struct:
 * \snippet Test_ContainerHelpers.cpp get_element_example_indexing_callable
 *
 * which would index the data structure in a manner in which the outer array
 * index varies fastest. The indexing function must take as arguments the
 * applicable container and a `size_t` index, in that order. This follows the
 * convention of `gsl::at`.
 * \note `std::complex` are regarded as non-indexable (despite a predictable
 * memory layout), so this function acts as the identity on `std::complex` of
 * fundamental types
 */
template <typename T, typename SubscriptFunction = GetContainerElement>
SPECTRE_ALWAYS_INLINE decltype(auto) get_element(
    T& t, const size_t i,
    SubscriptFunction at = GetContainerElement{}) noexcept {
  return ContainerHelpers_detail::GetImpls<(
      tt::is_complex_of_fundamental_v<std::remove_cv_t<T>> or
      cpp17::is_fundamental_v<std::remove_cv_t<T>>)>::get_element(t, i, at);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Retrieve the size of `t` if `t.size()` is a valid expression,
 * otherwise if `T` is fundamental or a `std::complex` of a fundamental type,
 * returns 1.
 *
 * \details This function also optionally takes the user-defined `size`
 * function, which can be used to specify a custom size function. For
 * instance, for a type which is a `std::array` of a `std::array`, the size
 * function could be the below callable struct:
 * \snippet Test_ContainerHelpers.cpp get_size_example_size_callable
 *
 * The `size` function must take the single argument of the applicable
 * container, and should return a `size_t`. This follows the convention of
 * `std::size()` as of C++17.
 * \note `std::complex` are regarded as non-indexable (despite a predictable
 * memory layout), so this function will return 1 for a `std::complex` of a
 * fundamental type
 */
template <typename T, typename SizeFunction = GetContainerSize>
SPECTRE_ALWAYS_INLINE decltype(auto) get_size(
    const T& t, SizeFunction size = GetContainerSize{}) noexcept {
  return ContainerHelpers_detail::GetImpls<(
      tt::is_complex_of_fundamental_v<std::remove_cv_t<T>> or
      cpp17::is_fundamental_v<std::remove_cv_t<T>>)>::get_size(t, size);
}
