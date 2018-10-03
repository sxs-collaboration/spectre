// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"  // IWYU pragma: keep
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/ReductionActions.hpp"   // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"               // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/IO/Observers/ObserverHelpers.hpp"

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace TestObservers_detail;

SPECTRE_TEST_CASE("Unit.IO.Observers.ReductionObserver", "[Unit][Observers]") {
  using TupleOfMockDistributedObjects =
      typename ActionTesting::MockRuntimeSystem<
          Metavariables>::TupleOfMockDistributedObjects;
  using obs_component = observer_component<Metavariables>;
  using obs_writer = observer_writer_component<Metavariables>;
  using element_comp = element_component<Metavariables>;

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using ObserverMockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          obs_component>;
  using WriterMockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          obs_writer>;
  using ElementMockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          element_comp>;
  TupleOfMockDistributedObjects dist_objects{};
  tuples::get<ObserverMockDistributedObjectsTag>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<obs_component>{});
  tuples::get<WriterMockDistributedObjectsTag>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<obs_writer>{});

  // Specific IDs have no significance, just need different IDs.
  const std::vector<ElementId<2>> element_ids{{1, {{{1, 0}, {1, 0}}}},
                                              {1, {{{1, 1}, {1, 0}}}},
                                              {1, {{{1, 0}, {2, 3}}}},
                                              {1, {{{1, 0}, {5, 4}}}},
                                              {0, {{{1, 0}, {1, 0}}}}};
  for (const auto& id : element_ids) {
    tuples::get<ElementMockDistributedObjectsTag>(dist_objects)
        .emplace(ElementIndex<2>{id},
                 ActionTesting::MockDistributedObject<element_comp>{});
  }

  tuples::TaggedTuple<observers::OptionTags::ReductionFileName,
                      observers::OptionTags::VolumeFileName>
      cache_data{};
  const auto& output_file_prefix =
      tuples::get<observers::OptionTags::ReductionFileName>(cache_data) =
          "./Unit.IO.Observers.ReductionObserver";
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      cache_data, std::move(dist_objects)};

  runner.simple_action<obs_component,
                       observers::Actions::Initialize<Metavariables>>(0);
  runner.simple_action<obs_writer,
                       observers::Actions::InitializeWriter<Metavariables>>(0);

  // Register elements
  for (const auto& id : element_ids) {
    runner.simple_action<element_comp,
                         observers::Actions::RegisterWithObservers<
                             observers::TypeOfObservation::Reduction>>(id, 0);
    // Invoke the simple_action RegisterSenderWithSelf that was called on the
    // observer component by the RegisterWithObservers action.
    runner.invoke_queued_simple_action<obs_component>(0);
  }

  const std::string h5_file_name = output_file_prefix + ".h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  using Redum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                         funcl::Sqrt<funcl::Divides<>>,
                                         std::index_sequence<1>>;
  using ReData = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<size_t, funcl::Plus<>>, Redum, Redum>;
  const auto make_fake_reduction_data = [](
      const observers::ArrayComponentId& id, const double time) noexcept {
    const auto hashed_id =
        static_cast<double>(std::hash<observers::ArrayComponentId>{}(id));
    constexpr size_t number_of_grid_points = 4;
    const double error0 = 1.0e-10 * hashed_id + time;
    const double error1 = 1.0e-12 * hashed_id + 2 * time;
    return ReData{time, number_of_grid_points, error0, error1};
  };

  const TimeId time(3);
  const std::vector<std::string> legend{"Time", "NumberOfPoints", "Error0",
                                        "Error1"};
  // Test passing reduction data.
  for (const auto& id : element_ids) {
    const observers::ArrayComponentId array_id(
        std::add_pointer_t<element_comp>{nullptr},
        Parallel::ArrayIndex<ElementIndex<2>>{ElementIndex<2>{id}});

    auto reduction_data_fakes =
        make_fake_reduction_data(array_id, time.value());
    runner.simple_action<obs_component,
                         observers::Actions::ContributeReductionData>(
        0, observers::ObservationId(time), legend,
        std::move(reduction_data_fakes));
  }
  // Invke the threaded action 'WriteReductionData' to write reduction data to
  // disk.
  runner.invoke_queued_threaded_action<obs_writer>(0);

  // Check that the H5 file was written correctly.
  {
    const auto file = h5::H5File<h5::AccessType::ReadOnly>(h5_file_name);
    const auto& dat_file = file.get<h5::Dat>("/element_data");
    const Matrix written_data = dat_file.get_data();
    const auto& written_legend = dat_file.get_legend();
    CHECK(written_legend == legend);
    const auto expected =
        alg::accumulate(
            element_ids, ReData(time.value(), 0, 0.0, 0.0),
            [&time, &make_fake_reduction_data ](
                ReData state, const ElementId<2>& id) noexcept {
              const observers::ArrayComponentId array_id(
                  std::add_pointer_t<element_comp>{nullptr},
                  Parallel::ArrayIndex<ElementIndex<2>>{ElementIndex<2>{id}});
              return state.combine(
                  make_fake_reduction_data(array_id, time.value()));
            })
            .finalize()
            .data();
    CHECK(std::get<0>(expected) == written_data(0, 0));
    CHECK(std::get<1>(expected) == written_data(0, 1));
    CHECK(std::get<2>(expected) == written_data(0, 2));
    CHECK(std::get<3>(expected) == written_data(0, 3));
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
