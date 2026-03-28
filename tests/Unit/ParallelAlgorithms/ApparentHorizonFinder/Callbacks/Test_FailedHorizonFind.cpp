// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/FailedHorizonFind.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Fr, bool Ignore>
struct HorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
  using temporal_id_tag = ::Tags::TimeAndPrevious<0>;
  using frame = Fr;

  using horizon_find_callbacks = tmpl::list<>;
  using horizon_find_failure_callbacks =
      tmpl::list<ah::callbacks::FailedHorizonFind<HorizonMetavars, Ignore>>;

  using compute_tags_on_element = tmpl::list<>;

  static constexpr ah::Destination destination = ah::Destination::ControlSystem;

  static std::string name() { return "TestingHorizonMetavars"; }
};

struct EmptyMetavars {
  using component_list = tmpl::list<>;
};

template <typename Fr>
void run_test() {
  const Parallel::GlobalCache<EmptyMetavars> cache{};

  const domain::creators::Sphere sphere_creator{
      0.9, 2.0, domain::creators::Sphere::Excision{nullptr}, 0_st, 4_st, true};
  const Domain<3> domain = sphere_creator.domain();

  const size_t l_max = 6;
  const LinkedMessageId<double> time{2.0, {1.0}};

  ah::Storage::Iteration<Fr> current_iteration{};
  current_iteration.compute_coords_retries = 2;
  current_iteration.strahlkorper =
      ylm::Strahlkorper<Fr>{l_max, 1.34, std::array{0.0, 0.0, 0.0}};
  current_iteration.block_coord_holders = ::block_logical_coordinates(
      domain, ylm::cartesian_coords(current_iteration.strahlkorper));

  // Manually set 4 points to be invalid
  for (size_t i = 0; i < 4; i++) {
    current_iteration.block_coord_holders.value()[i * 4].reset();
  }

  ah::Storage::SingleTimeStorage<Fr> current_time_storage{};
  current_time_storage.current_iteration = current_iteration;

  std::unordered_map<LinkedMessageId<double>,
                     ah::Storage::SingleTimeStorage<Fr>>
      all_storage{};
  all_storage[time] = current_time_storage;

  auto box = db::create<
      db::AddSimpleTags<tmpl::list<ah::Tags::CurrentTime, ah::Tags::Storage<Fr>,
                                   ah::Tags::FastFlow, ah::Tags::Verbosity>>>(
      std::optional{time}, all_storage,
      FastFlow{FastFlow::FlowType::Fast, 1.0, 0.5, 1.e-12, 1.e-2, 1.2, 5, 100},
      ::Verbosity::Quiet);

  static_assert(
      tt::assert_conforms_to_v<
          ah::callbacks::FailedHorizonFind<HorizonMetavars<Fr, true>, true>,
          ah::protocols::Callback>);
  static_assert(
      tt::assert_conforms_to_v<
          ah::callbacks::FailedHorizonFind<HorizonMetavars<Fr, false>, false>,
          ah::protocols::Callback>);

  // When we ignore the error, we just check that we can call things without an
  // error since things are only printed using printf.
  {
    ah::callbacks::FailedHorizonFind<HorizonMetavars<Fr, true>, true>::apply(
        box, cache, FastFlow::Status::MaxIts);
    ah::callbacks::FailedHorizonFind<HorizonMetavars<Fr, true>, true>::apply(
        box, cache, FastFlow::Status::InterpolationFailure);
  }
  // When erroring, check the message
  {
    CHECK_THROWS_WITH(
        (ah::callbacks::FailedHorizonFind<HorizonMetavars<Fr, false>, false>::
             apply(box, cache, FastFlow::Status::MaxIts)),
        Catch::Matchers::ContainsSubstring("TestingHorizonMetavars") and
            Catch::Matchers::ContainsSubstring("Too many iterations") and
            Catch::Matchers::ContainsSubstring("retries = 2") and
            not Catch::Matchers::ContainsSubstring("Invalid points"));
    CHECK_THROWS_WITH(
        (ah::callbacks::FailedHorizonFind<HorizonMetavars<Fr, false>, false>::
             apply(box, cache, FastFlow::Status::InterpolationFailure)),
        Catch::Matchers::ContainsSubstring("TestingHorizonMetavars") and
            Catch::Matchers::ContainsSubstring(
                "Cannot interpolate onto surface") and
            Catch::Matchers::ContainsSubstring("retries = 2") and
            Catch::Matchers::ContainsSubstring("Invalid points"));
  }
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.FailedHorizonFind",
                  "[ApparentHorizonFinder][Unit]") {
  domain::creators::register_derived_with_charm();
  run_test<Frame::Grid>();
  run_test<Frame::Inertial>();
}
}  // namespace
