// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "tests/Unit/TestingFramework.hpp"

void CkRegisterMainModule() {}

int main(int argc, char* argv[]) {
  // clang-tidy: internal warning in Catch
  return Catch::Session{}.run(argc, argv);  // NOLINT
}
