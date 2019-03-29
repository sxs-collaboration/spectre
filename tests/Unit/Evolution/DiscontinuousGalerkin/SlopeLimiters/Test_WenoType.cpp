// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/WenoType.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ExampleLimiterType {
  using type = SlopeLimiters::WenoType;
  static constexpr OptionString help = {"Example help text"};
};

void check_weno_type_parse(const std::string& input,
                           const SlopeLimiters::WenoType& expected_weno_type) {
  Options<tmpl::list<ExampleLimiterType>> opts("");
  opts.parse("ExampleLimiterType: " + input);
  CHECK(opts.get<ExampleLimiterType>() == expected_weno_type);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.WenoType",
                  "[SlopeLimiters][Unit]") {
  check_weno_type_parse("Hweno", SlopeLimiters::WenoType::Hweno);
  check_weno_type_parse("SimpleWeno", SlopeLimiters::WenoType::SimpleWeno);

  CHECK(get_output(SlopeLimiters::WenoType::Hweno) == "Hweno");
  CHECK(get_output(SlopeLimiters::WenoType::SimpleWeno) == "SimpleWeno");
}

// [[OutputRegex, Failed to convert "BadType" to WenoType]]
SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.WenoType.OptionParseError",
                  "[SlopeLimiters][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<ExampleLimiterType>> opts("");
  opts.parse("ExampleLimiterType: BadType");
  opts.get<ExampleLimiterType>();
}
