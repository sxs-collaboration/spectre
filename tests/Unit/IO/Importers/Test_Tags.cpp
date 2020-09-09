// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/TagName.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "IO/Importers/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"

namespace {
struct ExampleVolumeData {
  using group = importers::OptionTags::Group;
  static constexpr Options::String help = "Example volume data";
};
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Importers.Tags", "[Unit][IO]") {
  TestHelpers::db::test_simple_tag<importers::Tags::RegisteredElements>(
      "RegisteredElements");
  TestHelpers::db::test_simple_tag<
      importers::Tags::FileName<ExampleVolumeData>>(
      "FileName(ExampleVolumeData)");
  TestHelpers::db::test_simple_tag<
      importers::Tags::Subgroup<ExampleVolumeData>>(
      "Subgroup(ExampleVolumeData)");
  TestHelpers::db::test_simple_tag<
      importers::Tags::ObservationValue<ExampleVolumeData>>(
      "ObservationValue(ExampleVolumeData)");
  TestHelpers::db::test_simple_tag<importers::Tags::FunctionOfTimeFile>(
      "FunctionOfTimeFile");
  TestHelpers::db::test_simple_tag<importers::Tags::FunctionOfTimeNameMap>(
      "FunctionOfTimeNameMap");

  Options::Parser<
      tmpl::list<importers::OptionTags::FileName<ExampleVolumeData>,
                 importers::OptionTags::Subgroup<ExampleVolumeData>,
                 importers::OptionTags::ObservationValue<ExampleVolumeData>>>
      opts("");
  opts.parse(
      "Importers:\n"
      "  ExampleVolumeData:\n"
      "    FileName: File.name\n"
      "    Subgroup: data.group\n"
      "    ObservationValue: 1.");
  CHECK(opts.get<importers::OptionTags::FileName<ExampleVolumeData>>() ==
        "File.name");
  CHECK(opts.get<importers::OptionTags::Subgroup<ExampleVolumeData>>() ==
        "data.group");
  CHECK(
      opts.get<importers::OptionTags::ObservationValue<ExampleVolumeData>>() ==
      1.);
}
