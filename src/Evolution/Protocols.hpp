// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \ref protocols related to evolution systems
namespace evolution::protocols {

/*!
 *\brief Indicates the `ConformingType` represents the choice to start an
 * evolution with numeric initial data.
 *
 * Currently no requirements are imposed on the `ConformingType`.
 *
 * Here's an example of a class that conforms to this protocol:
 *
 * \snippet Evolution/Test_Protocols.cpp conforming_type_example
 */
struct NumericInitialData {
  template <typename ConformingType>
  struct test {};
};

}  // namespace evolution::protocols
