// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "RunTests.hpp"

#include "Framework/TestingFramework.hpp"

#include <charm++.h>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>

#include "Informer/InfoFromBuild.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/System/Abort.hpp"
#include "Utilities/System/Exit.hpp"

RunTests::RunTests(CkArgMsg* msg) {
  std::set_terminate(
      []() { sys::abort("Called terminate. Aborting..."); });
  Parallel::printf("%s", info_from_build().c_str());
  enable_floating_point_exceptions();
  const int result = Catch::Session().run(msg->argc, msg->argv);
  if (0 == result) {
    sys::exit();
  }
  sys::abort("A catch test has failed.");
}

#include "tests/Unit/RunTests.def.h"
