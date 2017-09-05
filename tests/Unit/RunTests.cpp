// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "tests/Unit/RunTests.hpp"

#include <catch.hpp>

#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Parallel/Abort.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/Printf.hpp"

RunTests::RunTests(CkArgMsg* msg) {
  std::set_terminate(
      []() { Parallel::abort("Called terminate. Aborting..."); });
  Parallel::printf("%s", info_from_build().c_str());
  enable_floating_point_exceptions();
  int result = Catch::Session().run(msg->argc, msg->argv);
  if (0 == result) {
    Parallel::exit();
  }
  Parallel::abort("A catch test has failed.");
}

#include "tests/Unit/RunTests.def.h"
