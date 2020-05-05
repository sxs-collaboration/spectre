// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace evolution {
/// \ref protocols related to evolution systems
namespace protocols {

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
struct NumericInitialData {
  template <typename ConformingType>
  struct test {
    using import_fields = typename ConformingType::import_fields;
  };
};

}  // namespace protocols
}  // namespace evolution
