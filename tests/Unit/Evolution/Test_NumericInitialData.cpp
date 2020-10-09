// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/NumericInitialData.hpp"
#include "Evolution/Protocols.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {

// Test that the numeric initial data class conforms to the protocol
static_assert(tt::assert_conforms_to<evolution::NumericInitialData,
                                     evolution::protocols::NumericInitialData>);

}  // namespace
