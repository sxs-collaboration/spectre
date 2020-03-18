// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/NoSuchType.hpp"
#include "Utilities/TypeTraits.hpp"

// @{
/*!
 * \ingroup TypeTraitsGroup
 * \brief Generate a type trait to check if a class has a `static constexpr`
 * variable, optionally also checking its type.
 *
 * To generate the corresponding `_v` metafunction, call
 * `CREATE_HAS_STATIC_MEMBER_VARIABLE` first and then
 * `CREATE_HAS_STATIC_MEMBER_VARIABLE_V`.
 *
 * \example
 * \snippet Test_CreateHasStaticMemberVariable.cpp CREATE_HAS_EXAMPLE
 *
 * \see `CREATE_IS_CALLABLE`
 */
// Use `NoSuchType*****` to represent any `VariableType`, i.e. not checking the
// variable type at all. If someone has that many pointers to a thing that isn't
// useful, it's their fault...
#define CREATE_HAS_STATIC_MEMBER_VARIABLE(CONSTEXPR_NAME)                    \
  template <typename CheckingType, typename VariableType = NoSuchType*****,  \
            typename = cpp17::void_t<>>                                      \
  struct has_##CONSTEXPR_NAME : std::false_type {};                          \
                                                                             \
  template <typename CheckingType, typename VariableType>                    \
  struct has_##CONSTEXPR_NAME<CheckingType, VariableType,                    \
                              cpp17::void_t<std::remove_const_t<decltype(    \
                                  CheckingType::CONSTEXPR_NAME)>>>           \
      : cpp17::bool_constant<                                                \
            cpp17::is_same_v<VariableType, NoSuchType*****> or               \
            cpp17::is_same_v<                                                \
                std::remove_const_t<decltype(CheckingType::CONSTEXPR_NAME)>, \
                VariableType>> {};

// Separate macros to avoid compiler warnings about unused variables
#define CREATE_HAS_STATIC_MEMBER_VARIABLE_V(CONSTEXPR_NAME)                 \
  template <typename CheckingType, typename VariableType = NoSuchType*****> \
  static constexpr const bool has_##CONSTEXPR_NAME##_v =                    \
      has_##CONSTEXPR_NAME<CheckingType, VariableType>::value;
// @}
