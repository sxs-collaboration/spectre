// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Events/RefineMesh.hpp"

#include <pup.h>

#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"

namespace amr::Events {
RefineMesh::RefineMesh() = default;

void RefineMesh::pup(PUP::er& p) { Event::pup(p); }

#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID RefineMesh::my_PUP_ID = 0;  // NOLINT
#endif                                        // SPECTRE_USE_CHARM
}  // namespace amr::Events
