// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/ProtocolHelpers.hpp"

namespace ProtocolHelpers_detail {

template <typename ConformingType, template <class> class Protocol>
struct TestProtocolConformanceImpl : std::true_type {
#ifdef SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
  static_assert(
      tt::conforms_to_v<ConformingType, Protocol>,
      "The type does not conform to the protocol or does not (publicly) "
      "inherit from it. The type is listed as the first template parameter to "
      "`test_protocol_conformance` and the protocol is listed as the second "
      "template parameter.");
#else   // SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
  static_assert(
      tt::conforms_to_v<ConformingType, Protocol>,
      "The type does not indicate it conforms to the protocol. The type is "
      "listed as the first template parameter to `test_protocol_conformance` "
      "and the protocol is listed as the second template parameter. "
      "Have you forgotten to (publicly) inherit it from the protocol?");
  static_assert(
      Protocol<ConformingType>::value,
      "The type does not conform to the protocol. The type is "
      "listed as the first template parameter to `test_protocol_conformance` "
      "and the protocol is listed as the second template parameter.");
#endif  // SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE
};

}  // namespace ProtocolHelpers_detail

/*!
 * \ingroup ProtocolsGroup
 * \brief Test that the `ConformingType` conforms to the `Protocol`
 *
 * Since the `tt::conforms_to_v` metafunction only checks if a class _indicates_
 * it conforms to the protocol (unless the
 * `SPECTRE_ALWAYS_CHECK_PROTOCOL_CONFORMANCE` flag is set), use the
 * `test_protocol_conformance` metafunction in unit tests to check the class
 * actually fulfills the requirements defined by the protocol's
 * `is_conforming_v` metafunction.
 */
template <typename ConformingType, template <class> class Protocol>
static constexpr bool test_protocol_conformance =
    ProtocolHelpers_detail::TestProtocolConformanceImpl<ConformingType,
                                                        Protocol>::value;
