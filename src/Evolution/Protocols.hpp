// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

namespace evolution {
/// \ref protocols related to evolution systems
namespace protocols {

namespace detail {
CREATE_HAS_TYPE_ALIAS(import_fields)
}  // namespace detail

/*!
 * \ingroup ProtocolsGroup
 * \brief Indicates the `ConformingType` provides compile-time information for
 * importing numeric initial data for an evolution.
 *
 * Requires the `ConformingType` has these type aliases:
 * - `import_fields`: The list of tags that should be imported from a volume
 * data file
 *
 * Here's an example of a class that conforms to this protocol:
 *
 * \snippet Evolution/Test_Protocols.cpp conforming_type_example
 */
template <typename ConformingType>
using NumericInitialData = detail::has_import_fields<ConformingType>;

}  // namespace protocols
}  // namespace evolution
