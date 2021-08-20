// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

/*!
 * \ingroup TypeTraitsGroup
 * \brief Generate a metafunction that will retrieve the specified type alias,
 * or if not present, assign a default type.
 *
 * \warning Please use defaults responsibly.  In many cases it is
 * better to require the type alias is present so the
 * functionality isn't hidden.  Also make sure to document the
 * compile-time interface, e.g. with a \ref protocols "protocol".
 */
#define CREATE_GET_TYPE_ALIAS_OR_DEFAULT(ALIAS_NAME)                           \
  template <typename CheckingType, typename Default, typename = std::void_t<>> \
  struct get_##ALIAS_NAME##_or_default {                                       \
    using type = Default;                                                      \
  };                                                                           \
  template <typename CheckingType, typename Default>                           \
  struct get_##ALIAS_NAME##_or_default<                                        \
      CheckingType, Default, std::void_t<typename CheckingType::ALIAS_NAME>> { \
    using type = typename CheckingType::ALIAS_NAME;                            \
  };                                                                           \
  template <typename CheckingType, typename Default>                           \
  using get_##ALIAS_NAME##_or_default_t =                                      \
      typename get_##ALIAS_NAME##_or_default<CheckingType, Default>::type;
