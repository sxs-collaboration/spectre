// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/Info.hpp"

#include <charm++.h>
#include <memory>

namespace Parallel {
Info::Info(CkMigrateMessage* msg) : ::sys::Info(msg) {}

auto Info::get_clone() const -> std::unique_ptr<::sys::Info> {
  return std::make_unique<Info>(*this);
}

void Info::pup(PUP::er& p) { ::sys::Info::pup(p); }

int Info::number_of_procs() const { return CkNumPes(); }

int Info::my_proc() const { return CkMyPe(); }

int Info::number_of_nodes() const { return CkNumNodes(); }

int Info::my_node() const { return CkMyNode(); }

int Info::procs_on_node([[maybe_unused]] const int node_index) const {
  return CkNodeSize(node_index);
}

int Info::my_local_rank() const { return CkMyRank(); }

int Info::first_proc_on_node([[maybe_unused]] const int node_index) const {
  return CkNodeFirst(node_index);
}

int Info::node_of([[maybe_unused]] const int proc_index) const {
  return CkNodeOf(proc_index);
}

int Info::local_rank_of([[maybe_unused]] const int proc_index) const {
  return CkRankOf(proc_index);
}

PUP::able::PUP_ID Info::my_PUP_ID = 0;
}  // namespace Parallel
