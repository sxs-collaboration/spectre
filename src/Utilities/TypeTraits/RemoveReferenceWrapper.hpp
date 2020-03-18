// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <functional>

#include "Utilities/TypeTraits.hpp"

namespace tt {
// @{
/*!
 * \ingroup TypeTraitsGroup
 * \brief Gets the underlying type if the type is a std::reference_wrapper,
 * otherwise returns the type itself
 *
 * \usage
 * For any type `I`,
 * \code
 * using result = tt::remove_reference_wrapper<I>;
 * \endcode
 * \metareturns
 * either `I::type` if `I` is a std::reference_wrapper, else returns I
 *
 * \example
 * \snippet Test_RemoveReferenceWrapper.cpp remove_reference_wrapper_example
 * \see std::reference_wrapper
 */
template <typename T>
struct remove_reference_wrapper {
  using type = T;
};

/// \cond HIDDEN_SYMBOLS
template <typename T>
struct remove_reference_wrapper<std::reference_wrapper<T>> {
  using type = T;
};

template <typename T>
struct remove_reference_wrapper<const std::reference_wrapper<T>> {
  using type = T;
};

template <typename T>
struct remove_reference_wrapper<volatile std::reference_wrapper<T>> {
  using type = T;
};

template <typename T>
struct remove_reference_wrapper<const volatile std::reference_wrapper<T>> {
  using type = T;
};
/// \endcond

template <typename T>
using remove_reference_wrapper_t = typename remove_reference_wrapper<T>::type;
// @}

// @{
/*!
 * \ingroup TypeTraitsGroup
 * \brief Removes std::reference_wrapper, references, and cv qualifiers.
 *
 * \example
 * \snippet Test_RemoveReferenceWrapper.cpp remove_cvref_wrap
 * \see std::reference_wrapper remove_reference_wrapper std::remove_cvref
 */
template <typename T>
struct remove_cvref_wrap {
  using type = cpp20::remove_cvref_t<tt::remove_reference_wrapper_t<T>>;
};

template <typename T>
using remove_cvref_wrap_t = typename remove_cvref_wrap<T>::type;
// @}
}  // namespace tt
