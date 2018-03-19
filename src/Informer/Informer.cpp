// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Informer/Informer.hpp"

#include <charm++.h>
#include <charm.h>

#include "Informer/InfoFromBuild.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/StdHelpers.hpp"

void Informer::print_startup_info(CkArgMsg* msg) {
  Parallel::printf(
      "\n"
      "Executing '%s' using %d processors.\n"
      "Date and time at startup: %s\n",
      msg->argv[0], Parallel::number_of_procs(),  // NOLINT
      current_date_and_time());

  Parallel::printf("%s\n", info_from_build());
}

void Informer::print_exit_info() {
  Parallel::printf(
      "\n"
      "Done!\n"
      "CkWallTimer in seconds %f\n"
      "Date and time at completion: %s\n",
      CkWallTimer(), current_date_and_time());
}
