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
#include "tests/Unit/RunTestsRegister.hpp"

RunTests::RunTests(CkArgMsg* msg) {
  std::set_terminate(
      []() { Parallel::abort("Called terminate. Aborting..."); });
  register_run_tests_libs();
  Parallel::printf("%s", info_from_build().c_str());
  enable_floating_point_exceptions();
  int result = Catch::Session().run(msg->argc, msg->argv);
  if (0 == result) {
    Parallel::exit();
  }
  Parallel::abort("A catch test has failed.");
}

#include "tests/Unit/RunTests.def.h"

// Needed for tests that use the ConstGlobalCache since it registers itself with
// Charm++. However, since Parallel/CharmMain.cpp isn't included in the RunTests
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
