// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/TagName.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "IO/Importers/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/String.hpp"

namespace {
struct ExampleVolumeData {
  static std::string name() { return "Example"; }
  static constexpr Options::String help = "Example volume data";
};
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Importers.Tags", "[Unit][IO]") {
  TestHelpers::db::test_simple_tag<importers::Tags::RegisteredElements<3>>(
      "RegisteredElements");
  TestHelpers::db::test_simple_tag<importers::Tags::ElementDataAlreadyRead>(
      "ElementDataAlreadyRead");
  TestHelpers::db::test_simple_tag<
      importers::Tags::ImporterOptions<ExampleVolumeData>>("VolumeData");

  Options::Parser<
      tmpl::list<importers::Tags::ImporterOptions<ExampleVolumeData>>>
      opts("");
  opts.parse(
      "Example:\n"
      "  VolumeData:\n"
      "    FileGlob: File.name\n"
      "    Subgroup: data.group\n"
      "    ObservationValue: 1.\n"
      "    Interpolate: True");
  const auto& options =
      opts.get<importers::Tags::ImporterOptions<ExampleVolumeData>>();
  using tuples::get;
  CHECK(get<importers::OptionTags::FileGlob>(options) == "File.name");
  CHECK(get<importers::OptionTags::Subgroup>(options) == "data.group");
  CHECK(std::get<double>(
            get<importers::OptionTags::ObservationValue>(options)) == 1.);
  CHECK(get<importers::OptionTags::EnableInterpolation>(options));

  CHECK(
      std::get<importers::ObservationSelector>(
          TestHelpers::test_option_tag<importers::OptionTags::ObservationValue>(
              "First")) == importers::ObservationSelector::First);
  CHECK(
      std::get<importers::ObservationSelector>(
          TestHelpers::test_option_tag<importers::OptionTags::ObservationValue>(
              "Last")) == importers::ObservationSelector::Last);
}
