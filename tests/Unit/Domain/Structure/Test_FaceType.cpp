// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/Structure/FaceType.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Structure.FaceType", "[Domain][Unit]") {
  CHECK(get_output(domain::FaceType::Uninitialized) == "Uninitialized");
  CHECK(get_output(domain::FaceType::External) == "External");
  CHECK(get_output(domain::FaceType::Topological) == "Topological");
  CHECK(get_output(domain::FaceType::ConformingAligned) == "ConformingAligned");
  CHECK(get_output(domain::FaceType::ConformingUnaligned) ==
        "ConformingUnaligned");
  CHECK(get_output(domain::FaceType::SingleNonconforming) ==
        "SingleNonconforming");
  CHECK(get_output(domain::FaceType::MultipleNonconforming) ==
        "MultipleNonconforming");
}
