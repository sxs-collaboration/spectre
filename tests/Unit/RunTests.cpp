// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "tests/Unit/RunTests.hpp"

#include <catch.hpp>

RunTests::RunTests(CkArgMsg* msg) {
  int result = Catch::Session().run(msg->argc, msg->argv);
  if (0 == result) {
    CkExit();
  }
  CkAbort("A catch test has failed.");
}

#include "tests/Unit/RunTests.def.h"
