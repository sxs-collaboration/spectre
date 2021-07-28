// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

/*!
 * \ingroup TypeTraitsGroup
 * \brief Generate a metafunction that will retrieve the specified
 * `static constexpr` member variable, or, if not present, a default.
 *
 * \note This macro requires the type traits created by
 * `CREATE_HAS_STATIC_MEMBER_VARIABLE` and
 * `CREATE_HAS_STATIC_MEMBER_VARIABLE_V` but does not create them --
 * both of those macros should be called before this macro to generate
 * the needed type traits.
 */
#define CREATE_GET_STATIC_MEMBER_VARIABLE_OR_DEFAULT(ALIAS_NAME)          \
  template <typename CheckingType, auto Default,                          \
            bool present = has_##ALIAS_NAME##_v<CheckingType>>            \
  struct get_##ALIAS_NAME##_or_default {                                  \
    static constexpr auto value = Default;                                \
  };                                                                      \
  template <typename CheckingType, auto Default>                          \
  struct get_##ALIAS_NAME##_or_default<CheckingType, Default, true> {     \
    static_assert(                                                        \
        std::is_same_v<std::decay_t<decltype(Default)>,                   \
                       std::decay_t<decltype(CheckingType::ALIAS_NAME)>>, \
        "Types of default and found value are not the same.");            \
    static constexpr auto value = CheckingType::ALIAS_NAME;               \
  };                                                                      \
  template <typename CheckingType, auto Default>                          \
  constexpr auto get_##ALIAS_NAME##_or_default_v =                        \
      get_##ALIAS_NAME##_or_default<CheckingType, Default>::value;
