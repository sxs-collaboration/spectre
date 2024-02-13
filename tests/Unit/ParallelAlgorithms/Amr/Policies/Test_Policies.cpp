// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <sstream>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "ParallelAlgorithms/Amr/Policies/Isotropy.hpp"
#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"

namespace {
void test_equality() {
  INFO("Equality");
  CHECK(amr::Policies{amr::Isotropy::Anisotropic} ==
        amr::Policies{amr::Isotropy::Anisotropic});
  CHECK(amr::Policies{amr::Isotropy::Anisotropic} !=
        amr::Policies{amr::Isotropy::Isotropic});
}

void test_pup() {
  INFO("Serialization");
  test_serialization(amr::Policies{amr::Isotropy::Anisotropic});
}

void test_option_parsing() {
  INFO("Option Parsing creation");
  const auto isotropies =
      std::array{amr::Isotropy::Anisotropic, amr::Isotropy::Isotropic};
  for (const auto isotropy : isotropies) {
    std::ostringstream creation_string;
    creation_string << "Isotropy: " << isotropy;
    const auto policies =
        TestHelpers::test_creation<amr::Policies>(creation_string.str());
    CHECK(policies == amr::Policies{isotropy});
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Amr.Policies",
                  "[ParallelAlgorithms][Unit]") {
  test_equality();
  test_pup();
  test_option_parsing();
}
