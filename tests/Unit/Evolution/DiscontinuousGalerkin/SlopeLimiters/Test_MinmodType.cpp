// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ExampleLimiterType {
  using type = SlopeLimiters::MinmodType;
  static constexpr OptionString help = {"Example help text"};
};

void check_minmod_type_parse(
    const std::string& input,
    const SlopeLimiters::MinmodType& expected_minmod_type) {
  Options<tmpl::list<ExampleLimiterType>> opts("");
  opts.parse("ExampleLimiterType: " + input);
  CHECK(opts.get<ExampleLimiterType>() == expected_minmod_type);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.MinmodType",
                  "[SlopeLimiters][Unit]") {
  check_minmod_type_parse("LambdaPi1", SlopeLimiters::MinmodType::LambdaPi1);
  check_minmod_type_parse("LambdaPiN", SlopeLimiters::MinmodType::LambdaPiN);
  check_minmod_type_parse("Muscl", SlopeLimiters::MinmodType::Muscl);

  CHECK(get_output(SlopeLimiters::MinmodType::LambdaPi1) == "LambdaPi1");
  CHECK(get_output(SlopeLimiters::MinmodType::LambdaPiN) == "LambdaPiN");
  CHECK(get_output(SlopeLimiters::MinmodType::Muscl) == "Muscl");
}

// [[OutputRegex, Failed to convert "BadType" to MinmodType]]
SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.MinmodType.OptionParseError",
                  "[SlopeLimiters][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<ExampleLimiterType>> opts("");
  opts.parse("ExampleLimiterType: BadType");
  opts.get<ExampleLimiterType>();
}
