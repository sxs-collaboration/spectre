// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <pup.h>
#include <string>
#include <tuple>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/Events/ObserveFields.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace {
using ObserveFields = Events::ObserveFields;
constexpr size_t l_max = 3;
constexpr size_t num_radial_grid_points = 2;
constexpr size_t num_angular_grid_points =
    Spectral::Swsh::number_of_swsh_collocation_points(l_max);
constexpr size_t num_volume_grid_points =
    num_radial_grid_points * num_angular_grid_points;
constexpr double time = 1.3;

struct MockWriteReductionDataRow {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& /*box*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& file_legend,
                    const std::tuple<std::vector<double>>& data_row) {
    // If we were to have made the data non-zero values, the event does too many
    // transformations for us to calculate what the result would be by hand, so
    // we just check that the sizes are correct, and that the (code) time is
    // correct
    const size_t l_plus_one_squared = square(l_max + 1);
    if (subfile_name.find("InertialRetardedTime") != std::string::npos) {
      CHECK(file_legend.size() == l_plus_one_squared + 1);
      CHECK(std::get<0>(data_row).size() == l_plus_one_squared + 1);
    } else if (subfile_name.find("OneMinusY") != std::string::npos) {
      CHECK(file_legend.size() == num_radial_grid_points + 1);
      CHECK(std::get<0>(data_row).size() == num_radial_grid_points + 1);
    } else {
      CHECK(subfile_name.find("CompactifiedRadius") != std::string::npos);
      CHECK(file_legend.size() == 2 * l_plus_one_squared + 1);
      CHECK(std::get<0>(data_row).size() == 2 * l_plus_one_squared + 1);
    }
    CHECK(std::get<0>(data_row)[0] == time);
  }
};

template <typename Metavariables>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<>>>>>;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;

  using replace_these_threaded_actions =
      tmpl::list<observers::ThreadedActions::WriteReductionDataRow>;
  using with_these_threaded_actions = tmpl::list<MockWriteReductionDataRow>;
};

template <typename Metavariables>
struct MockElement {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<
          tmpl::push_back<ObserveFields::available_tags_to_observe, Tags::LMax,
                          Tags::NumberOfRadialPoints, ::Tags::Time>>>>>;
};

struct Metavars {
  void pup(PUP::er& /*p*/) {}
  using observed_reduction_data_tags = tmpl::list<>;
  using component_list =
      tmpl::list<MockObserverWriter<Metavars>, MockElement<Metavars>>;
};

void test() {
  using metavars = Metavars;
  using obs_writer = MockObserverWriter<metavars>;
  using element = MockElement<metavars>;

  Cce::Events::ObserveFields fields{std::vector<std::string>{
      "InertialRetardedTime", "J", "Dy(H)", "OneMinusY"}};
  Cce::Events::ObserveFields serialized_fields =
      serialize_and_deserialize(fields);

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_nodegroup_component_and_initialize<obs_writer>(
      make_not_null(&runner), {});
  ActionTesting::emplace_singleton_component_and_initialize<element>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, {});

  auto& cache = ActionTesting::cache<obs_writer>(runner, 0);
  auto& box = ActionTesting::get_databox<element>(make_not_null(&runner), 0);
  const int array_index = 0;
  const obs_writer* const component = nullptr;
  const Event::ObservationValue observation_value{};

  // We only set the sizes of the tensors that we are using
  const auto set_number = [&box](const auto tag_v, const auto value_to_set_to) {
    using tag = std::decay_t<decltype(tag_v)>;
    db::mutate<tag>(
        [&value_to_set_to](
            const gsl::not_null<typename tag::type*> value_to_set) {
          *value_to_set = value_to_set_to;
        },
        make_not_null(&box));
  };
  const auto size_data = [&box](auto tag_v, const size_t size) {
    using tag = std::decay_t<decltype(tag_v)>;
    db::mutate<tag>(
        [&size](const auto tensor) {
          // All are scalars
          get(*tensor) = ComplexDataVector{size, 0.0};
        },
        make_not_null(&box));
  };
  set_number(Tags::LMax{}, l_max);
  set_number(Tags::NumberOfRadialPoints{}, num_radial_grid_points);
  set_number(::Tags::Time{}, time);
  size_data(Tags::ComplexInertialRetardedTime{}, num_angular_grid_points);
  size_data(Tags::BondiJ{}, num_volume_grid_points);
  size_data(Tags::Dy<Tags::BondiH>{}, num_volume_grid_points);
  size_data(Tags::OneMinusY{}, num_volume_grid_points);

  serialized_fields(box, cache, array_index, component, observation_value);

  // 1 for InertialRetardedTime and OneMinusY and 2 for each other tag
  const size_t expected_number_of_actions = 6;
  CHECK(ActionTesting::number_of_queued_threaded_actions<obs_writer>(
            runner, 0) == expected_number_of_actions);
  for (size_t i = 0; i < expected_number_of_actions; i++) {
    ActionTesting::invoke_queued_threaded_action<obs_writer>(
        make_not_null(&runner), 0);
  }
}

void test_errors() {
  CHECK_THROWS_WITH((Cce::Events::ObserveFields{std::vector<std::string>{
                        "Unique", "Duplicate", "Duplicate"}}),
                    Catch::Matchers::ContainsSubstring(
                        "more than once in list of variables to observe"));

  CHECK_THROWS_WITH(
      (Cce::Events::ObserveFields{std::vector<std::string>{"MisspelledTag"}}),
      Catch::Matchers::ContainsSubstring(
          "MisspelledTag is not an available variable. Available variables:"));
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Events.ObserveFields",
                  "[Unit][Evolution]") {
  test();
  test_errors();
}
}  // namespace
}  // namespace Cce
