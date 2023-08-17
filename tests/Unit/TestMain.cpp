// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch2/catch_session.hpp>
#include <catch2/catch_tostring.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <cstddef>
#include <memory>

#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

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
