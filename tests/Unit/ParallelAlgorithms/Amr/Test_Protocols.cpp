// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  // [amr_projectors]
  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using projectors =
        tmpl::list<Initialization::ProjectTimeStepping<1>,
                   evolution::dg::Initialization::ProjectDomain<1>>;
  };
  // [amr_projectors]
};
static_assert(tt::assert_conforms_to_v<Metavariables::amr,
                                       ::amr::protocols::AmrMetavariables>);
}  // namespace
