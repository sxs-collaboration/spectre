// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <type_traits>

#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"
#include "Utilities/TypeTraits/IsComplexOfFundamental.hpp"

namespace tt {
namespace detail {
// NOLINTBEGIN(clang-diagnostic-unused-const-variable)
CREATE_HAS_TYPE_ALIAS(ElementType)
CREATE_HAS_TYPE_ALIAS_V(ElementType)
CREATE_HAS_TYPE_ALIAS(value_type)
CREATE_HAS_TYPE_ALIAS_V(value_type)
// NOLINTEND(clang-diagnostic-unused-const-variable)
}  // namespace detail
/// @{
/// \ingroup TypeTraitsGroup
/// \brief Extracts the fundamental type for a container
///
/// \details  Designates a type alias `get_fundamental_type::type`
///  as `T` when `T` itself is an appropriate fundamental type, and the
/// contained type of a container which specifies a `value_type`.
///
/// `get_fundamental_type_t<T>` is provided as a type alias to
/// `type` from `get_fundamental_type<T>`
///
/// \snippet Test_GetFundamentalType.cpp get_fundamental_type
template <typename T, typename = std::nullptr_t>
struct get_fundamental_type {
  using type = tmpl::conditional_t<std::is_fundamental_v<T>, T, NoSuchType>;
};

/// \cond
// Specialization for Blaze expressions
template <typename T>
struct get_fundamental_type<T, Requires<detail::has_ElementType_v<T>>> {
  using type = typename get_fundamental_type<typename T::ElementType>::type;
};
// Specialization for containers
template <typename T>
struct get_fundamental_type<T, Requires<detail::has_value_type_v<T> and
                                        not detail::has_ElementType_v<T>>> {
  using type = typename get_fundamental_type<typename T::value_type>::type;
};
/// \endcond

template <typename T>
using get_fundamental_type_t = typename get_fundamental_type<T>::type;
/// @}

template <typename T, typename = std::nullptr_t>
struct get_complex_or_fundamental_type {
  using type =
      std::conditional_t<tt::is_complex_or_fundamental_v<T>, T, NoSuchType>;
};

/// \cond
// Specialization for Blaze expressions
template <typename T>
struct get_complex_or_fundamental_type<T,
                                       Requires<detail::has_ElementType_v<T>>> {
  using type =
      typename get_complex_or_fundamental_type<typename T::ElementType>::type;
};
// Specialization for containers
template <typename T>
struct get_complex_or_fundamental_type<
    T, Requires<detail::has_value_type_v<T> and
                not detail::has_ElementType_v<T> and
                not tt::is_complex_or_fundamental_v<T>>> {
  using type =
      typename get_complex_or_fundamental_type<typename T::value_type>::type;
};
/// \endcond

template <typename T>
using get_complex_or_fundamental_type_t =
    typename get_complex_or_fundamental_type<T>::type;

}  // namespace tt
