// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "tests/Unit/RunTests.hpp"

#include "tests/Unit/TestingFramework.hpp"

#include <charm++.h>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>

#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Parallel/Abort.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/Printf.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/RunTestsRegister.hpp"

RunTests::RunTests(CkArgMsg* msg) {
  std::set_terminate(
      []() { Parallel::abort("Called terminate. Aborting..."); });
  register_run_tests_libs();
  Parallel::printf("%s", info_from_build().c_str());
  enable_floating_point_exceptions();
  const int result = Catch::Session().run(msg->argc, msg->argv);
  // In the case where we run all the non-failure tests at once we must ensure
  // that we only initialize and finalize the python env once. Initialization is
  // done in the constructor of SetupLocalPythonEnvironment, while finalization
  // is done in the constructor of RunTests.
  pypp::SetupLocalPythonEnvironment::finalize_env();
  if (0 == result) {
    Parallel::exit();
  }
  Parallel::abort("A catch test has failed.");
}

#include "tests/Unit/RunTests.def.h"  /// IWYU pragma: keep

// Needed for tests that use the ConstGlobalCache since it registers itself with
// Charm++. However, since Parallel/CharmMain.tpp isn't included in the RunTests
// executable, no actual registration is done, the ConstGlobalCache is only
// queued for registration.
namespace Parallel {
namespace charmxx {
class RegistrationHelper;
/// \cond
std::unique_ptr<RegistrationHelper>* charm_register_list = nullptr;
size_t charm_register_list_capacity = 0;
size_t charm_register_list_size = 0;
/// \endcond
}  // namespace charmxx
}  // namespace Parallel
