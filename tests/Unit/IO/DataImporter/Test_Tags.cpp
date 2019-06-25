// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "IO/DataImporter/Tags.hpp"

SPECTRE_TEST_CASE("Unit.IO.DataImporter.Tags", "[Unit][IO]") {
  CHECK(db::tag_name<importer::Tags::RegisteredElements>() ==
        "RegisteredElements");
}
