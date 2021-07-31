// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

/*!
 * \ingroup TypeTraitsGroup
 * \brief Generate a metafunction that will retrieve the specified type alias,
 * or if not present, assign a default type.
 *
 * \note This macro requires the type traits created by `CREATE_HAS_TYPE_ALIAS`
 * and `CREATE_HAS_TYPE_ALIAS_V` but does not create them -- both of those
 * macros should be called before this macro to generate the needed type traits.
 *
 * \warning Please use defaults responsibly.  In many cases it is
 * better to require the static member variable is present so the
 * functionality isn't hidden.  Also make sure to document the
 * compile-time interface, e.g. with a \ref protocols "protocol".
 */
#define CREATE_GET_TYPE_ALIAS_OR_DEFAULT(ALIAS_NAME)                  \
  template <typename CheckingType, typename Default,                  \
            bool present = has_##ALIAS_NAME##_v<CheckingType>>        \
  struct get_##ALIAS_NAME##_or_default {                              \
    using type = Default;                                             \
  };                                                                  \
  template <typename CheckingType, typename Default>                  \
  struct get_##ALIAS_NAME##_or_default<CheckingType, Default, true> { \
    using type = typename CheckingType::ALIAS_NAME;                   \
  };                                                                  \
  template <typename CheckingType, typename Default>                  \
  using get_##ALIAS_NAME##_or_default_t =                             \
      typename get_##ALIAS_NAME##_or_default<CheckingType, Default>::type;
