// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "IO/Observer/Tags.hpp"
#include "Utilities/TypeTraits.hpp"

namespace observers::Tags {
SPECTRE_TEST_CASE("Unit.IO.Observers.Tags", "[Unit][Observers]") {
  TestHelpers::db::test_simple_tag<ExpectedContributorsForObservations>(
      "ExpectedContributorsForObservations");
  TestHelpers::db::test_simple_tag<ContributorsOfReductionData>(
      "ContributorsOfReductionData");
  TestHelpers::db::test_simple_tag<NodesThatContributedReductions>(
      "NodesThatContributedReductions");
  TestHelpers::db::test_simple_tag<NodesExpectedToContributeReductions>(
      "NodesExpectedToContributeReductions");
  TestHelpers::db::test_simple_tag<ContributorsOfTensorData>(
      "ContributorsOfTensorData");
  TestHelpers::db::test_simple_tag<TensorData>("TensorData");
  TestHelpers::db::test_simple_tag<ReductionData<double>>("ReductionData");
  TestHelpers::db::test_simple_tag<ReductionDataNames<double>>(
      "ReductionDataNames");
  TestHelpers::db::test_simple_tag<H5FileLock>("H5FileLock");
  TestHelpers::db::test_simple_tag<VolumeFileName>("VolumeFileName");
  TestHelpers::db::test_simple_tag<ReductionFileName>("ReductionFileName");
  TestHelpers::db::test_simple_tag<NodesExpectedToContributeReductions>(
      "NodesExpectedToContributeReductions");
  static_assert(
      std::is_same_v<typename ReductionData<double, int, char>::names_tag,
                     ReductionDataNames<double, int, char>>,
      "Failed testing Observers tags");
  static_assert(
      std::is_same_v<typename ReductionDataNames<double, int, char>::data_tag,
                     ReductionData<double, int, char>>,
      "Failed testing Observers tags");
}
}  // namespace observers::Tags
