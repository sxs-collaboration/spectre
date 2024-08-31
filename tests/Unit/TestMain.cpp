// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch2/catch_session.hpp>
#include <catch2/catch_tostring.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <cstddef>
#include <memory>

#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/MemoryHelpers.hpp"

class TestRunListener : public Catch::EventListenerBase {
 public:
  using Catch::EventListenerBase::EventListenerBase;

  void fatalErrorEncountered(Catch::StringRef error) { ERROR(error); }
};

CATCH_REGISTER_LISTENER(TestRunListener)

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-declarations"
extern "C" void CkRegisterMainModule(void) {}
#pragma GCC diagnostic pop

int main(int argc, char* argv[]) {
  setup_error_handling();
  setup_memory_allocation_failure_reporting();
  Parallel::printf("%s", info_from_build().c_str());
  enable_floating_point_exceptions();
  enable_segfault_handler();
  Catch::StringMaker<double>::precision =
      std::numeric_limits<double>::max_digits10;
  Catch::StringMaker<float>::precision =
      std::numeric_limits<float>::max_digits10;

  const int result = Catch::Session().run(argc, argv);

  // Clean up the Python environment only after all tests have finished running,
  // since there could be multiple tests run in a single executable launch.
  pypp::SetupLocalPythonEnvironment::finalize_env();

  return result;
}
