// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <type_traits>

#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits/IsComplexOfFundamental.hpp"

/// \ingroup UtilitiesGroup
/// \brief Callable struct which retrieves the `t.size()` for operand `t`. This
/// will cause a compiler error if no such function exists.
struct GetContainerSize {
  template <typename T>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(const T& t) const noexcept {
    return t.size();
  }
};

/// \ingroup UtilitiesGroup
/// \brief Callable struct for the subscript operator. Returns `t[i]`
struct GetContainerElement {
  template <typename T>
  SPECTRE_ALWAYS_INLINE decltype(auto) operator()(T& t, const size_t i) const
      noexcept {
    return t[i];
  }
};

/// \ingroup UtilitiesGroup
/// \brief Callable struct which applies the `t.destructive_resize()` for
/// operand `t`. This will cause a compiler error if no such function exists.
struct ContainerDestructiveResize {
  template <typename T>
  SPECTRE_ALWAYS_INLINE void operator()(T& t, const size_t size) const
      noexcept {
    return t.destructive_resize(size);
  }
};

namespace ContainerHelpers_detail {
// implementation struct for get_element and get_size
template <bool IsFundamentalOrComplexOfFundamental>
struct ContainerImpls;

template <>
struct ContainerImpls<true> {
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

  template <typename T, typename DestructiveResizeFunction>
  static SPECTRE_ALWAYS_INLINE void apply_destructive_resize(
      T& /*t*/, const size_t /*i*/,
      const DestructiveResizeFunction /*destructive_resize*/) noexcept {
    // no-op for fundamental types.
  }
};

template <>
struct ContainerImpls<false> {
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

  template <typename T, typename DestructiveResizeFunction>
  static SPECTRE_ALWAYS_INLINE void apply_destructive_resize(
      T& t, const size_t size,
      const DestructiveResizeFunction destructive_resize) noexcept {
    destructive_resize(t, size);
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
  return ContainerHelpers_detail::ContainerImpls<(
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
  return ContainerHelpers_detail::ContainerImpls<(
      tt::is_complex_of_fundamental_v<std::remove_cv_t<T>> or
      cpp17::is_fundamental_v<std::remove_cv_t<T>>)>::get_size(t, size);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Checks the size of each component of the container, and resizes if
 * necessary.
 *
 * \details This operation is not permitted when any of the components of the
 * tensor is non-owning (see `VectorImpl` for ownership details).
 * \note This utility should NOT be used when it is anticipated that the
 * components will be the wrong size. In that case, suggest either manual
 * checking or restructuring so that resizing is less common. The internal
 * call uses `UNLIKELY` to perform the checks most quickly when resizing is
 * unnecessary.
 *
 * \warning This is typically to be called on `Tensor`s, NOT (for instance)
 * `DataVector`s. For derived classes of `VectorImpl`, this function will cause
 * no resize. Instead, use the `VectorImpl` member function
 * `VectorImpl::destructive_resize()`.
 *
 * \note This assumes that a range-based iterator will appropriately loop over
 * the elements to resize, and that each resized element is either a fundamental
 * type or a derived class of `VectorImpl`. If either of those assumptions needs
 * to be relaxed, this function will need to be generalized.
 */
template <typename Container,
          typename DestructiveResizeFunction = ContainerDestructiveResize>
void SPECTRE_ALWAYS_INLINE destructive_resize_components(
    const gsl::not_null<Container*> container, const size_t new_size,
    DestructiveResizeFunction destructive_resize =
        ContainerDestructiveResize{}) noexcept {
  for (auto& vector : *container) {
    ContainerHelpers_detail::ContainerImpls<(
        tt::is_complex_of_fundamental_v<
            std::remove_cv_t<typename Container::value_type>> or
        cpp17::is_fundamental_v<
            std::remove_cv_t<typename Container::value_type>>)>::
        apply_destructive_resize(vector, new_size, destructive_resize);
  }
}
