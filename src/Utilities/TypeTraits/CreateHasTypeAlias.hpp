// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/NoSuchType.hpp"
#include "Utilities/TypeTraits.hpp"

// @{
/*!
 * \ingroup TypeTraitsGroup
 * \brief Generate a type trait to check if a class has a type alias with a
 * particular name, optionally also checking its type.
 *
 * To generate the corresponding `_v` metafunction, call
 * `CREATE_HAS_TYPE_ALIAS` first and then `CREATE_HAS_TYPE_ALIAS_V`.
 *
 * \example
 * \snippet Test_CreateHasTypeAlias.cpp CREATE_HAS_TYPE_ALIAS
 *
 * \see `CREATE_IS_CALLABLE`
 */
// Use `NoSuchType*****` to represent any `AliasType`, i.e. not checking the
// alias type at all. If someone has that many pointers to a thing that isn't
// useful, it's their fault...
#define CREATE_HAS_TYPE_ALIAS(ALIAS_NAME)                                     \
  template <typename CheckingType, typename AliasType = NoSuchType*****,      \
            typename = cpp17::void_t<>>                                       \
  struct has_##ALIAS_NAME : std::false_type {};                               \
                                                                              \
  template <typename CheckingType, typename AliasType>                        \
  struct has_##ALIAS_NAME<CheckingType, AliasType,                            \
                          cpp17::void_t<typename CheckingType::ALIAS_NAME>>   \
      : cpp17::bool_constant<                                                 \
            cpp17::is_same_v<AliasType, NoSuchType*****> or                   \
            cpp17::is_same_v<typename CheckingType::ALIAS_NAME, AliasType>> { \
  };
// Separate macros to avoid compiler warnings about unused variables
#define CREATE_HAS_TYPE_ALIAS_V(ALIAS_NAME)                              \
  template <typename CheckingType, typename AliasType = NoSuchType*****> \
  static constexpr const bool has_##ALIAS_NAME##_v =                     \
      has_##ALIAS_NAME<CheckingType, AliasType>::value;
// @}
