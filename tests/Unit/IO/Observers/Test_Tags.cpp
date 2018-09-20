// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "IO/Observer/Tags.hpp"

namespace observers {
namespace Tags {
SPECTRE_TEST_CASE("Unit.IO.Observers.Tags", "[Unit][Observers]") {
  CHECK(NumberOfEvents::name() == "NumberOfEvents");
  CHECK(ReductionArrayComponentIds::name() == "ReductionArrayComponentIds");
  CHECK(VolumeArrayComponentIds::name() == "VolumeArrayComponentIds");
  CHECK(TensorData::name() == "TensorData");
  CHECK(VolumeDataLock::name() == "VolumeDataLock");
  CHECK(ReductionDataLock::name() == "ReductionDataLock");
}
}  // namespace Tags
}  // namespace observers
