// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "IO/Observer/Tags.hpp"

SPECTRE_TEST_CASE("Unit.IO.Observers.Tags", "[Unit][Observers]") {
  CHECK(observers::Tags::NumberOfEvents::name() == "NumberOfEvents");
  CHECK(observers::Tags::ReductionArrayComponentIds::name() ==
        "ReductionArrayComponentIds");
  CHECK(observers::Tags::VolumeArrayComponentIds::name() ==
        "VolumeArrayComponentIds");
  CHECK(observers::Tags::TensorData::name() == "TensorData");
}
