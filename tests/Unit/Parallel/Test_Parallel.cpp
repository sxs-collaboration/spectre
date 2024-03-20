// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Utilities/System/ParallelInfo.hpp"

SPECTRE_TEST_CASE("Unit.Parallel.NodeAndPes", "[Unit][Parallel]") {
  CHECK(1 == sys::number_of_procs());
  CHECK(0 == sys::my_proc());
  CHECK(1 == sys::number_of_nodes());
  CHECK(0 == sys::my_node());
  CHECK(1 == sys::procs_on_node(sys::my_node()));
  CHECK(0 == sys::my_local_rank());
  CHECK(0 == sys::first_proc_on_node(sys::my_node()));
  CHECK(0 == sys::local_rank_of(sys::my_proc()));
  CHECK(0 == sys::node_of(sys::my_proc()));
  // We check that the wall time is greater than or equal to zero and less
  // than 2 seconds, just to check the function actually returns something.
  const double walltime = sys::wall_time();
  CHECK((0 <= walltime and 2 >= walltime));
}
