// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "TestMainCharm.hpp"

#include "Framework/TestingFramework.hpp"

#include <charm++.h>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <string>

#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/System/Abort.hpp"
#include "Utilities/System/Exit.hpp"

TestMainCharm::TestMainCharm(CkArgMsg* msg) {
  setup_error_handling();
  setup_memory_allocation_failure_reporting();
  Parallel::printf("%s", info_from_build().c_str());
  enable_floating_point_exceptions();
  enable_segfault_handler();
  Catch::StringMaker<double>::precision =
      std::numeric_limits<double>::max_digits10;
  Catch::StringMaker<float>::precision =
      std::numeric_limits<float>::max_digits10;
  const int result = Catch::Session().run(msg->argc, msg->argv);
  // Clean up the Python environment only after all tests have finished running,
  // since there could be multiple tests run in a single executable launch.
  pypp::SetupLocalPythonEnvironment::finalize_env();
  if (0 == result) {
    sys::exit();
  }
  sys::abort("A catch test has failed.");
}

#include "TestMainCharm.def.h"
