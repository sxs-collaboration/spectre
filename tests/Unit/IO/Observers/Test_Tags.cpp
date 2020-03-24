// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "IO/Observer/Tags.hpp"
#include "Utilities/TypeTraits.hpp"

namespace observers {
namespace Tags {
SPECTRE_TEST_CASE("Unit.IO.Observers.Tags", "[Unit][Observers]") {
  TestHelpers::db::test_simple_tag<NumberOfEvents>("NumberOfEvents");
  TestHelpers::db::test_simple_tag<ReductionArrayComponentIds>(
      "ReductionArrayComponentIds");
  TestHelpers::db::test_simple_tag<VolumeArrayComponentIds>(
      "VolumeArrayComponentIds");
  TestHelpers::db::test_simple_tag<TensorData>("TensorData");
  TestHelpers::db::test_simple_tag<VolumeObserversRegistered>(
      "VolumeObserversRegistered");
  TestHelpers::db::test_simple_tag<VolumeObserversContributed>(
      "VolumeObserversContributed");
  TestHelpers::db::test_simple_tag<H5FileLock>("H5FileLock");
  TestHelpers::db::test_simple_tag<ReductionData<double>>("ReductionData");
  TestHelpers::db::test_simple_tag<ReductionDataNames<double>>(
      "ReductionDataNames");
  TestHelpers::db::test_simple_tag<ReductionObserversContributed>(
      "ReductionObserversContributed");
  TestHelpers::db::test_simple_tag<ReductionObserversRegistered>(
      "ReductionObserversRegistered");
  TestHelpers::db::test_simple_tag<ReductionObserversRegisteredNodes>(
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
