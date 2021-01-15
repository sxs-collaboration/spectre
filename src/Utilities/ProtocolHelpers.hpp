// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits.hpp"

namespace tt {

/*!
 * \ingroup ProtocolsGroup
 * \brief Indicate a class conforms to the `Protocol`.
 *
 * (Publicly) inherit classes from this class to indicate they conform to the
 * `Protocol`.
 *
 * \see Documentation on \ref protocols
 */
template <typename Protocol>
struct ConformsTo {};

// Note that std::is_convertible is used in the following type aliases as it
// will not match private base classes (unlike std::is_base_of)

// @{
/*!
 * \ingroup ProtocolsGroup
 * \brief Checks if the `ConformingType` conforms to the `Protocol`.
 *
 * This metafunction is SFINAE-friendly. See `tt::assert_conforms_to` for a
 * metafunction that is not SFINAE-friendly but that triggers static asserts
 * with diagnostic messages to understand why the `ConformingType` does not
 * conform to the `Protocol`.
 *
 * This metafunction only checks if the class derives off the protocol to reduce
 * compile time. Protocol conformance is tested rigorously in the unit tests
 * instead.
 *
 * \see Documentation on \ref protocols
 * \see tt::assert_conforms_to
 */
template <typename ConformingType, typename Protocol>
using conforms_to =
    typename std::is_convertible<ConformingType*, ConformsTo<Protocol>*>;
template <typename ConformingType, typename Protocol>
constexpr bool conforms_to_v =
    std::is_convertible_v<ConformingType*, ConformsTo<Protocol>*>;
// @}

namespace detail {

template <typename ConformingType, typename Protocol>
struct AssertConformsToImpl : std::true_type {
  static_assert(
      tt::conforms_to_v<ConformingType, Protocol>,
      "The type does not indicate it conforms to the protocol. The type is "
      "listed as the first template parameter to `assert_conforms_to` "
      "and the protocol is listed as the second template parameter. "
      "Have you forgotten to (publicly) inherit the type from "
      "tt::ConformsTo<Protocol>?");
  // Implicitly instantiate Protocol::test in order to test conformance
  static_assert(
      not std::is_same_v<
          decltype(typename Protocol::template test<ConformingType>{}), void>);
};

}  // namespace detail

/*!
 * \ingroup ProtocolsGroup
 * \brief Assert that the `ConformingType` conforms to the `Protocol`.
 *
 * Similar to `tt::conforms_to`, but not SFINAE-friendly. Instead, triggers
 * static asserts with diagnostic messages to understand why the
 * `ConformingType` fails to conform to the `Protocol`.
 *
 * \see Documentation on \ref protocols
 * \see tt::conforms_to
 */
template <typename ConformingType, typename Protocol>
static constexpr bool assert_conforms_to =
    detail::AssertConformsToImpl<ConformingType, Protocol>::value;

}  // namespace tt
