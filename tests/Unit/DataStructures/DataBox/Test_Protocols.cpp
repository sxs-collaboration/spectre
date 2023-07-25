// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "Helpers/DataStructures/DataBox/Examples.hpp"
#include "Utilities/ProtocolHelpers.hpp"

static_assert(tt::assert_conforms_to_v<db::TestHelpers::ExampleMutator,
                                       db::protocols::Mutator>);
static_assert(
    tt::assert_conforms_to_v<db::TestHelpers::ExampleCreatedFromOptionsTagAdder,
                             db::protocols::CreatedFromOptionsTagAdder>);
