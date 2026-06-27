// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/LinkedMessageId.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Initialization.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

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
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.Initialization",
                  "[ApparentHorizonFinder][Unit]") {
  (void)MockHorizonMetavars::destination;

  const FastFlow expected_fast_flow{
      FastFlow::FlowType::Fast, 1.0, 0.5, 1.e-12, 1.e-2, 1.2, 5, 100};
  const ah::HorizonOptions<Frame::Grid> horizon_options{
      std::vector<std::unique_ptr<ah::Criterion>>{},
      ylm::Strahlkorper<Frame::Grid>{4, 2.0, std::array{0.0, 0.0, 0.0}},
      expected_fast_flow,
      ::Verbosity::Debug,
      3,
      std::nullopt};

  std::optional<size_t> current_resolution_l{4};
  ::Verbosity verbosity{};
  std::optional<LinkedMessageId<double>> current_time =
      LinkedMessageId<double>{1.0, std::nullopt};
  FastFlow fast_flow{};

  ah::Initialize<MockHorizonMetavars>::apply(
      make_not_null(&current_resolution_l), make_not_null(&verbosity),
      make_not_null(&fast_flow), make_not_null(&current_time), horizon_options);

  CHECK(verbosity == ::Verbosity::Debug);
  CHECK_FALSE(current_time.has_value());
  CHECK(fast_flow == expected_fast_flow);
  CHECK_FALSE(current_resolution_l.has_value());
}
