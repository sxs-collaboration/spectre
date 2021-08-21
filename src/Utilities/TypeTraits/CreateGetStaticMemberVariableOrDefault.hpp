// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

/*!
 * \ingroup TypeTraitsGroup
 * \brief Generate a metafunction that will retrieve the specified
 * `static constexpr` member variable, or, if not present, a default.
 *
 * \warning Please use defaults responsibly.  In many cases it is
 * better to require the static member variable is present so the
 * functionality isn't hidden.  Also make sure to document the
 * compile-time interface, e.g. with a \ref protocols "protocol".
 */
#define CREATE_GET_STATIC_MEMBER_VARIABLE_OR_DEFAULT(VARIABLE_NAME)          \
  template <typename CheckingType, auto Default, typename = std::void_t<>>   \
  struct get_##VARIABLE_NAME##_or_default {                                  \
    static constexpr auto value = Default;                                   \
  };                                                                         \
  template <typename CheckingType, auto Default>                             \
  struct get_##VARIABLE_NAME##_or_default<                                   \
      CheckingType, Default,                                                 \
      std::void_t<decltype(CheckingType::VARIABLE_NAME)>> {                  \
    static_assert(                                                           \
        std::is_same_v<std::decay_t<decltype(Default)>,                      \
                       std::decay_t<decltype(CheckingType::VARIABLE_NAME)>>, \
        "Types of default and found value are not the same.");               \
    static constexpr auto value = CheckingType::VARIABLE_NAME;               \
  };                                                                         \
  template <typename CheckingType, auto Default>                             \
  constexpr auto get_##VARIABLE_NAME##_or_default_v =                        \
      get_##VARIABLE_NAME##_or_default<CheckingType, Default>::value;
