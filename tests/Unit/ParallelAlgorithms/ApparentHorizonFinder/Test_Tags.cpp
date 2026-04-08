// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {
struct MockHorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
  using time_tag = ::Tags::TimeAndPrevious<0>;

  using frame = ::Frame::Grid;

  // Don't need callbacks
  using horizon_find_callbacks = tmpl::list<>;
  using horizon_find_failure_callbacks = tmpl::list<>;

  using compute_tags_on_element = tmpl::list<>;

  static constexpr ah::Destination destination = ah::Destination::ControlSystem;

  static std::string name() { return "MockHorizonMetavars"; }
};

void test_observation_time() {
  LinkedMessageId<double> result{};
  ah::Tags::ObservationTimeCompute<0>::function(make_not_null(&result), 1.2);
  CHECK(result == LinkedMessageId<double>{1.2, std::nullopt});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.Tags",
                  "[ApparentHorizonFinder][Unit]") {
  (void)MockHorizonMetavars::destination;

  TestHelpers::db::test_simple_tag<ah::Tags::CurrentResolutionL>(
      "CurrentResolutionL");
  TestHelpers::db::test_simple_tag<ah::Tags::Verbosity>("Verbosity");
  TestHelpers::db::test_simple_tag<ah::Tags::FastFlow>("FastFlow");
  TestHelpers::db::test_simple_tag<ah::Tags::CurrentTime>("CurrentTime");
  TestHelpers::db::test_simple_tag<ah::Tags::PendingTimes>("PendingTimes");
  TestHelpers::db::test_simple_tag<ah::Tags::CompletedTimes>("CompletedTimes");
  TestHelpers::db::test_simple_tag<ah::Tags::Storage<::Frame::Distorted>>(
      "Storage");
  TestHelpers::db::test_simple_tag<
      ah::Tags::PreviousSurfaces<::Frame::Distorted>>("PreviousSurfaces");
  TestHelpers::db::test_simple_tag<
      ah::Tags::ApparentHorizonOptions<MockHorizonMetavars>>(
      "ApparentHorizonOptions");
  TestHelpers::db::test_simple_tag<ah::Tags::BlocksForHorizonFind>(
      "BlocksForHorizonFind");
  TestHelpers::db::test_simple_tag<ah::Tags::ObservationTime<0>>(
      "AhObservationTime0");
  TestHelpers::db::test_compute_tag<ah::Tags::ObservationTimeCompute<0>>(
      "AhObservationTime0");
  test_observation_time();
  TestHelpers::db::test_simple_tag<ah::Tags::ObserveCenters>("ObserveCenters");
  TestHelpers::db::test_simple_tag<ah::Tags::LMax>("LMax");
}
