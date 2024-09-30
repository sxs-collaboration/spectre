// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Events/RefineMesh.hpp"

#include <pup.h>

#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"

namespace amr::Events {
RefineMesh::RefineMesh() = default;

RefineMesh::RefineMesh(CkMigrateMessage* m) : Event(m) {}

void RefineMesh::pup(PUP::er& p) { Event::pup(p); }

PUP::able::PUP_ID RefineMesh::my_PUP_ID = 0;  // NOLINT
}  // namespace amr::Events
