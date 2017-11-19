// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines metafunction make_boost_variant_over

#pragma once

#include <boost/variant.hpp>

#include "Utilities/TypeTraits.hpp"

namespace detail {
template <typename Sequence>
struct make_boost_variant_over_impl;

template <template <typename...> class Sequence, typename... Ts>
struct make_boost_variant_over_impl<Sequence<Ts...>> {
  static_assert(not cpp17::disjunction<std::is_same<
                    std::decay_t<std::remove_pointer_t<Ts>>, void>...>::value,
                "Cannot create a boost::variant with a 'void' type.");
  using type = boost::variant<Ts...>;
};
}  // namespace detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Create a boost::variant with all all the types inside the typelist
 * Sequence
 *
 * \metareturns boost::variant of all types inside `Sequence`
 */
template <typename Sequence>
using make_boost_variant_over =
    typename detail::make_boost_variant_over_impl<Sequence>::type;
