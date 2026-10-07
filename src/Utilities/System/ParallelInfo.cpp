// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/System/ParallelInfo.hpp"

#include <charm++.h>

namespace sys {
int number_of_procs() { return CkNumPes(); }

int my_proc() { return CkMyPe(); }

int number_of_nodes() { return CkNumNodes(); }

int my_node() { return CkMyNode(); }

int procs_on_node([[maybe_unused]] const int node_index) {
  return CkNodeSize(node_index);
}

int my_local_rank() { return CkMyRank(); }

int first_proc_on_node([[maybe_unused]] const int node_index) {
  return CkNodeFirst(node_index);
}

int node_of([[maybe_unused]] const int proc_index) {
  return CkNodeOf(proc_index);
}

int local_rank_of([[maybe_unused]] const int proc_index) {
  return CkRankOf(proc_index);
}

double wall_time() { return CkWallTimer(); }
}  // namespace sys
