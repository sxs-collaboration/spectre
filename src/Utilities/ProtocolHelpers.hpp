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
template <template <class> class Protocol>
struct ConformsTo {};

// Note that std::is_convertible is used in the following type aliases as it
// will not match private base classes (unlike std::is_base_of)

// @{
/*!
 * \ingroup ProtocolsGroup
 * \brief Checks if the `ConformingType` conforms to the `Protocol`.
 *
 * By default, only checks if the class derives off the protocol to reduce
 * compile time. Protocol conformance is tested rigorously in the unit tests
 * instead. Set the `SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE` CMake option to
 * always enable rigorous protocol conformance checks.
 *
 * \see Documentation on \ref protocols
 */
#ifdef SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
template <typename ConformingType, template <class> class Protocol>
constexpr bool conforms_to_v =
    cpp17::is_convertible_v<ConformingType*, ConformsTo<Protocol>*>and
        Protocol<ConformingType>::value;
template <typename ConformingType, template <class> class Protocol>
using conforms_to =
    cpp17::bool_constant<conforms_to_v<ConformingType, Protocol>>;
#else   // SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
template <typename ConformingType, template <class> class Protocol>
using conforms_to =
    typename std::is_convertible<ConformingType*, ConformsTo<Protocol>*>;
template <typename ConformingType, template <class> class Protocol>
constexpr bool conforms_to_v =
    cpp17::is_convertible_v<ConformingType*, ConformsTo<Protocol>*>;
#endif  // SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
// @}

}  // namespace tt
