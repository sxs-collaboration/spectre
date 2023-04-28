// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Informer/Informer.hpp"

#include <charm++.h>
#include <charm.h>
#include <sstream>

#include "Informer/InfoFromBuild.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/System/ParallelInfo.hpp"

void Informer::print_startup_info(CkArgMsg* msg) {
  std::stringstream ss{};
  for (int i = 0; i < msg->argc - 1; i++) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    ss << msg->argv[i] << " ";
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  ss << msg->argv[msg->argc - 1];

  Parallel::printf(
      "\n"
      "Executing '%s' using %d processors.\n"
      "Launch command line: %s\n"
      "Charm++ startup time in seconds: %f\n"
      "Date and time at startup: %s\n",
      executable_name(), sys::number_of_procs(),
      ss.str(),  // NOLINT
      sys::wall_time(), current_date_and_time());

  Parallel::printf("%s\n", info_from_build());
}

void Informer::print_exit_info() {
  Parallel::printf(
      "\n"
      "Done!\n"
      "Wall time: %s\n"
      "Date and time at completion: %s\n",
      sys::pretty_wall_time(), current_date_and_time());
}
