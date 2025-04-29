// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/LinkedMessageId.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/SendDependencyToObserverWriter.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <bool WriteVolData>
struct HorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
  using temporal_id_tag = ::Tags::TimeAndPrevious<0>;
  using frame = Frame::Inertial;

  using horizon_find_callbacks = tmpl::list<>;
  using horizon_find_failure_callbacks = tmpl::list<>;

  using compute_tags_on_element = tmpl::list<>;

  static constexpr ah::Destination destination = ah::Destination::Observation;

  static std::string name() { return "TestingHorizonMetavars"; }
};

template <bool WriteVolData>
struct MockContributeDependency {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const double time, const std::string& dependency,
                    std::string volume_subfile_name,
                    const bool write_volume_data) {
    CHECK(time == 2.0);
    CHECK(volume_subfile_name == "FakeDependency");
    CHECK(dependency == "TestingHorizonMetavars");
    CHECK(write_volume_data == WriteVolData);
  }
};

template <typename Metavariables, bool WriteVolData>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::list<observers::Tags::ReductionFileName,
                                             observers::Tags::SurfaceFileName>;
  using simple_tags =
      typename observers::Actions::InitializeWriter<Metavariables>::simple_tags;
  using compute_tags = typename observers::Actions::InitializeWriter<
      Metavariables>::compute_tags;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<observers::Actions::InitializeWriter<Metavariables>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked = observers::ObserverWriter<Metavariables>;

  using replace_these_threaded_actions =
      tmpl::list<observers::ThreadedActions::ContributeDependency>;
  using with_these_threaded_actions =
      tmpl::list<MockContributeDependency<WriteVolData>>;
};

template <bool WriteVolData>
struct Metavariables {
  using observed_reduction_data_tags = tmpl::list<>;
  using component_list =
      tmpl::list<MockObserverWriter<Metavariables, WriteVolData>>;
};

template <bool WriteVolData>
void run_test() {
  CAPTURE(WriteVolData);

  using metavars = Metavariables<WriteVolData>;
  using obs_writer = MockObserverWriter<metavars, WriteVolData>;
  ActionTesting::MockRuntimeSystem<metavars> runner{{}};
  ActionTesting::emplace_nodegroup_component<obs_writer>(
      make_not_null(&runner));

  auto& cache = ActionTesting::cache<obs_writer>(runner, 0_st);

  const LinkedMessageId<double> time{2.0, {1.0}};

  auto box =
      db::create<db::AddSimpleTags<ah::Tags::CurrentTime, ah::Tags::Dependency,
                                   ah::Tags::Verbosity>>(
          std::optional{time},
          WriteVolData ? std::optional{"FakeDependency"s} : std::nullopt,
          ::Verbosity::Silent);

  ah::callbacks::SendDependencyToObserverWriter<
      HorizonMetavars<WriteVolData>,
      WriteVolData>::apply(box, cache, FastFlow::Status::TruncationTol);

  REQUIRE(ActionTesting::number_of_queued_threaded_actions<obs_writer>(
              runner, 0_st) == (WriteVolData ? 1 : 0));

  if constexpr (WriteVolData) {
    ActionTesting::invoke_queued_threaded_action<obs_writer>(
        make_not_null(&runner), 0_st);
  }
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.SendDependencyToObserverWriter",
                  "[ApparentHorizonFinder][Unit]") {
  run_test<true>();
  run_test<false>();
}
}  // namespace
