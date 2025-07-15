// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/LinkedMessageId.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/CleanUp.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
namespace {
template <typename Fr>
void test_cleanup() {
  LinkedMessageId<double> expected_current_time{3.0, {2.0}};
  std::optional<LinkedMessageId<double>> current_time{expected_current_time};
  std::unordered_map<LinkedMessageId<double>,
                     ah::Storage::SingleTimeStorage<Fr>>
      all_storage{};
  all_storage[expected_current_time] = ah::Storage::SingleTimeStorage<Fr>{};
  std::set<LinkedMessageId<double>> completed_times{
      LinkedMessageId<double>{1.0, std::nullopt},
      LinkedMessageId<double>{2.0, {1.0}}};
  FastFlow fast_flow{
      FastFlow::FlowType::Fast, 1.0, 0.5, 1.e-12, 1.e-2, 1.2, 5, 100};

  clean_up_horizon_finder(
      make_not_null(&current_time), make_not_null(&all_storage),
      make_not_null(&completed_times), make_not_null(&fast_flow));

  CHECK_FALSE(current_time.has_value());
  CHECK_FALSE(all_storage.contains(expected_current_time));
  CHECK(completed_times.contains(expected_current_time));
  CHECK(fast_flow.current_iteration() == 0);

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      clean_up_horizon_finder(
          make_not_null(&current_time), make_not_null(&all_storage),
          make_not_null(&completed_times), make_not_null(&fast_flow)),
      Catch::Matchers::ContainsSubstring(
          "Current time must be set in order to clean up the horizon finder"));
#endif

  // Check that completed times is limited to 1000 entries
  expected_current_time = LinkedMessageId<double>{2000.0, {1999.0}};
  current_time = expected_current_time;
  all_storage[expected_current_time] = ah::Storage::SingleTimeStorage<Fr>{};
  for (size_t i = 4; i < 2000; i++) {
    completed_times.insert(LinkedMessageId<double>{
        static_cast<double>(i), {static_cast<double>(i - 1)}});
  }

  clean_up_horizon_finder(
      make_not_null(&current_time), make_not_null(&all_storage),
      make_not_null(&completed_times), make_not_null(&fast_flow));

  CHECK(completed_times.size() == 1000);
  CHECK(*std::prev(completed_times.end()) == expected_current_time);
  CHECK(*completed_times.begin() == LinkedMessageId<double>{1001.0, {1000.0}});
  CHECK(fast_flow.current_iteration() == 0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.CleanUp",
                  "[ApparentHorizonFinder][Unit]") {
  test_cleanup<::Frame::Grid>();
  test_cleanup<::Frame::Distorted>();
  test_cleanup<::Frame::Inertial>();
}
}  // namespace ah
