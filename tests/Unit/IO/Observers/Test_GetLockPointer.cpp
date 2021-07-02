// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Observer/Actions/GetLockPointer.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
Parallel::NodeLock* h5_lock_to_check;
Parallel::NodeLock* volume_lock_to_check;

template <typename LockTag>
struct mock_lock_retrieval_action {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, typename ArrayIndex>
  static void apply(const db::DataBox<DbTagList>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) noexcept {
    auto lock = Parallel::get_parallel_component<
                    observers::ObserverWriter<Metavariables>>(cache)
                    .ckLocalBranch()
                    ->template local_synchronous_action<
                        observers::Actions::GetLockPointer<LockTag>>();
    if constexpr (std::is_same_v<LockTag, observers::Tags::H5FileLock>) {
      h5_lock_to_check = lock;
    } else {
      volume_lock_to_check = lock;
    }
  }
};

template <typename Metavariables>
struct mock_observer_writer {
  using chare_type = ActionTesting::MockNodeGroupChare;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using simple_tags =
      tmpl::list<observers::Tags::H5FileLock, observers::Tags::VolumeDataLock>;

  using const_global_cache_tags = tmpl::list<>;

  using metavariables = Metavariables;
  using array_index = size_t;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <typename Metavariables>
struct mock_array {
  using chare_type = ActionTesting::MockArrayChare;
  using replace_these_simple_actions = tmpl::list<>;
  using with_these_simple_actions = tmpl::list<>;

  using const_global_cache_tags = tmpl::list<>;

  using metavariables = Metavariables;
  using array_index = size_t;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

struct test_metavariables {
  using component_list = tmpl::list<mock_observer_writer<test_metavariables>,
                                    mock_array<test_metavariables>>;
  enum class Phase { Initialization, Evolve, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Observer.GetNodeLockPointer", "[Unit][Cce]") {
  using array_component = mock_array<test_metavariables>;
  using writer_component = mock_observer_writer<test_metavariables>;

  ActionTesting::MockRuntimeSystem<test_metavariables> runner{{}};

  ActionTesting::set_phase(make_not_null(&runner),
                           test_metavariables::Phase::Initialization);
  ActionTesting::emplace_array_component_and_initialize<writer_component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0,
      {Parallel::NodeLock{}, Parallel::NodeLock{}});
  ActionTesting::emplace_array_component<array_component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0);

  ActionTesting::simple_action<
      array_component, mock_lock_retrieval_action<observers::Tags::H5FileLock>>(
      make_not_null(&runner), 0);
  ActionTesting::simple_action<
      array_component,
      mock_lock_retrieval_action<observers::Tags::VolumeDataLock>>(
      make_not_null(&runner), 0);

  db::mutate<observers::Tags::H5FileLock, observers::Tags::VolumeDataLock>(
      make_not_null(
          &ActionTesting::get_databox<
              writer_component, tmpl::list<observers::Tags::H5FileLock,
                                           observers::Tags::VolumeDataLock>>(
              make_not_null(&runner), 0)),
      [](const gsl::not_null<Parallel::NodeLock*> h5_lock,
         const gsl::not_null<Parallel::NodeLock*> volume_lock) noexcept {
        CHECK(h5_lock.get() == h5_lock_to_check);
        CHECK(volume_lock.get() == volume_lock_to_check);
      });
}
