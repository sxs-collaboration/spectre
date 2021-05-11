// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Protocols/Metavariables.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {

// [domain_metavariables_example]
struct DomainMetavariables : tt::ConformsTo<domain::protocols::Metavariables> {
  static constexpr bool enable_time_dependent_maps = false;
};
// [domain_metavariables_example]

static_assert(tt::assert_conforms_to<DomainMetavariables,
                                     domain::protocols::Metavariables>);
}  // namespace
