// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct TestGroup {
  static constexpr Options::String help = "halp";
};

void test_construct_from_options() {
  Options::Parser<tmpl::list<logging::OptionTags::Verbosity<TestGroup>>> opts(
      "");
  opts.parse("TestGroup:\n  Verbosity: Verbose\n");
  CHECK(opts.get<logging::OptionTags::Verbosity<TestGroup>>() ==
        Verbosity::Verbose);
}

void test_construct_from_options_fail() {
  Options::Parser<tmpl::list<logging::OptionTags::Verbosity<TestGroup>>> opts(
      "");
  opts.parse("TestGroup:\n  Verbosity: Braggadocious\n");  // Meant to fail.
  CHECK_THROWS_WITH((opts.get<logging::OptionTags::Verbosity<TestGroup>>()),
                    Catch::Matchers::Contains(
                        "Failed to convert \"Braggadocious\" to Verbosity"));
}

void test_ostream() {
  CHECK(get_output(Verbosity::Silent) == "Silent");
  CHECK(get_output(Verbosity::Quiet) == "Quiet");
  CHECK(get_output(Verbosity::Verbose) == "Verbose");
  CHECK(get_output(Verbosity::Debug) == "Debug");
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Logging.Verbosity", "[Unit]") {
  test_construct_from_options();
  test_ostream();
  test_construct_from_options_fail();
}
