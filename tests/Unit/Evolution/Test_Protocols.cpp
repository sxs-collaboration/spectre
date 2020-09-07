// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Protocols.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {

// [conforming_type_example]
struct ValidNumericInitialData
    : tt::ConformsTo<evolution::protocols::NumericInitialData> {};
// [conforming_type_example]

static_assert(tt::assert_conforms_to<ValidNumericInitialData,
                                     evolution::protocols::NumericInitialData>);

}  // namespace
