// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Elliptic/Systems/ScalarGaussBonnet/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.ScalarGaussBonnet.Tags",
                  "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<sgb::Tags::Epsilon2>("Epsilon2");
  TestHelpers::db::test_simple_tag<sgb::Tags::Epsilon4>("Epsilon4");
  TestHelpers::db::test_simple_tag<sgb::Tags::RolloffLocation>(
      "RolloffLocation");
  TestHelpers::db::test_simple_tag<sgb::Tags::RolloffRate>("RolloffRate");
  TestHelpers::db::test_simple_tag<sgb::Tags::Psi>("Psi");
  TestHelpers::db::test_simple_tag<sgb::Tags::RolledOffShift>("RolledOffShift");
  TestHelpers::db::test_simple_tag<sgb::Tags::PiWithRolledOffShift>(
      "PiWithRolledOffShift");
  CHECK(TestHelpers::test_option_tag<sgb::OptionTags::Epsilon2>("31.01") ==
        31.01);
  CHECK(TestHelpers::test_option_tag<sgb::OptionTags::Epsilon4>("5.08") ==
        5.08);
  CHECK(TestHelpers::test_option_tag<sgb::OptionTags::RolloffLocation>(
            "4.05") == 4.05);
  CHECK(TestHelpers::test_option_tag<sgb::OptionTags::RolloffRate>("28.1") ==
        28.1);
}
