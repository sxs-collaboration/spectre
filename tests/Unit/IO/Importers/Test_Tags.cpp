// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "IO/Importers/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"

namespace {
struct ExampleVolumeData {
  using group = importers::OptionTags::Group;
  static constexpr OptionString help = "Example volume data";
};
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Importers.Tags", "[Unit][IO]") {
  CHECK(db::tag_name<importers::Tags::RegisteredElements>() ==
        "RegisteredElements");
  CHECK(db::tag_name<importers::Tags::FileName<ExampleVolumeData>>() ==
        "FileName(ExampleVolumeData)");
  CHECK(db::tag_name<importers::Tags::Subgroup<ExampleVolumeData>>() ==
        "Subgroup(ExampleVolumeData)");
  CHECK(db::tag_name<importers::Tags::ObservationValue<ExampleVolumeData>>() ==
        "ObservationValue(ExampleVolumeData)");

  Options<
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
