// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/ParallelAlgorithms/ApparentHorizonFinder/TestHelpers.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace TestHelpers::ah {
static_assert(tt::assert_conforms_to_v<ExampleHorizonFindCallback,
                                       ::ah::protocols::Callback>);

static_assert(tt::assert_conforms_to_v<ExampleHorizonMetavars,
                                       ::ah::protocols::HorizonMetavars>);
}  // namespace TestHelpers::ah
