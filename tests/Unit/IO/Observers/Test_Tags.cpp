// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "IO/Observer/Tags.hpp"
#include "Utilities/TypeTraits.hpp"

namespace observers {
namespace Tags {
SPECTRE_TEST_CASE("Unit.IO.Observers.Tags", "[Unit][Observers]") {
  CHECK(NumberOfEvents::name() == "NumberOfEvents");
  CHECK(ReductionArrayComponentIds::name() == "ReductionArrayComponentIds");
  CHECK(VolumeArrayComponentIds::name() == "VolumeArrayComponentIds");
  CHECK(TensorData::name() == "TensorData");
  CHECK(VolumeObserversRegistered::name() == "VolumeObserversRegistered");
  CHECK(VolumeObserversContributed::name() == "VolumeObserversContributed");
  CHECK(H5FileLock::name() == "H5FileLock");
  CHECK(ReductionData<double>::name() == "ReductionData");
  CHECK(ReductionDataNames<double>::name() == "ReductionDataNames");
  CHECK(ReductionObserversContributed::name() ==
        "ReductionObserversContributed");
  CHECK(ReductionObserversRegistered::name() == "ReductionObserversRegistered");
  CHECK(ReductionObserversRegisteredNodes::name() ==
        "ReductionObserversRegisteredNodes");
  static_assert(
      cpp17::is_same_v<typename ReductionData<double, int, char>::names_tag,
                       ReductionDataNames<double, int, char>>,
      "Failed testing Observers tags");
  static_assert(
      cpp17::is_same_v<typename ReductionDataNames<double, int, char>::data_tag,
                       ReductionData<double, int, char>>,
      "Failed testing Observers tags");
}
}  // namespace Tags
}  // namespace observers
