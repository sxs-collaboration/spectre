// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"

namespace tt {
// @{
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
template <typename T, typename Enable = std::void_t<>>
struct get_fundamental_type {
  using type = tmpl::conditional_t<std::is_fundamental_v<T>, T, NoSuchType>;
};

/// \cond
template <typename T>
struct get_fundamental_type<T, std::void_t<typename T::value_type>> {
  using type = typename get_fundamental_type<typename T::value_type>::type;
};
/// \endcond

template <typename T>
using get_fundamental_type_t = typename get_fundamental_type<T>::type;
// @}
}  // namespace tt
