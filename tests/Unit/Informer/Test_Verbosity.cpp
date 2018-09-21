// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {

void test_construct_from_options() noexcept {
  Options<tmpl::list<OptionTags::Verbosity>> opts("");
  opts.parse("Verbosity: Verbose\n");
  CHECK(opts.get<OptionTags::Verbosity>() == Verbosity::Verbose);
}

void test_construct_from_options_fail() noexcept {
  Options<tmpl::list<OptionTags::Verbosity>> opts("");
  opts.parse("Verbosity: Braggadocious\n");  // Meant to fail.
  opts.get<OptionTags::Verbosity>();
}

void test_ostream() noexcept {
  CHECK(get_output(Verbosity::Silent) == "Silent");
  CHECK(get_output(Verbosity::Quiet) == "Quiet");
  CHECK(get_output(Verbosity::Verbose) == "Verbose");
  CHECK(get_output(Verbosity::Debug) == "Debug");
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Informer.Verbosity", "[Informer][Unit]") {
  test_construct_from_options();
  test_ostream();
}

// [[OutputRegex, Failed to convert "Braggadocious" to Verbosity]]
SPECTRE_TEST_CASE("Unit.Informer.Verbosity.Fail", "[Informer][Unit]") {
  ERROR_TEST();
  test_construct_from_options_fail();
}
