// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <set>
#include <unordered_map>

#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RotationAboutZAxis.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Component.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/CurrentTime.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace ah {
namespace {
void test_set_current_time() {
  std::optional<LinkedMessageId<double>> current_time =
      LinkedMessageId<double>{1.0, std::nullopt};
  std::set<LinkedMessageId<double>> pending_times{};
  std::set<LinkedMessageId<double>> completed_times{};
  std::unordered_map<LinkedMessageId<double>,
                     ah::Storage::SingleTimeStorage<Frame::Grid>>
      all_storage{};
  const std::unordered_map<ElementId<3>, Storage::VolumeVariables<Frame::Grid>>
      unused_volume_vars{};
  const Storage::Iteration<Frame::Grid> unused_interation{};
  const ylm::Strahlkorper<Frame::Grid> unused_prev_strahlkorper{};
  const ::Verbosity verbosity = ::Verbosity::Silent;
  const std::string name{"HorizonMetavars"};

  const auto add_to_all_storage = [&](const LinkedMessageId<double>& time,
                                      const Destination destination) {
    all_storage.emplace(time, ah::Storage::SingleTimeStorage<Frame::Grid>{
                                  unused_volume_vars,
                                  {},
                                  unused_interation,
                                  unused_prev_strahlkorper,
                                  destination});
  };

  // Current time has value, so do nothing
  {
    set_current_time(make_not_null(&current_time),
                     make_not_null(&pending_times), completed_times,
                     all_storage, verbosity, name);
    CHECK(current_time ==
          std::optional{LinkedMessageId<double>{1.0, std::nullopt}});
    CHECK(pending_times.empty());
  }

  // Current time doesn't have a value, but pending times is empty, so do
  // nothing
  {
    current_time.reset();
    set_current_time(make_not_null(&current_time),
                     make_not_null(&pending_times), completed_times,
                     all_storage, verbosity, name);
    CHECK_FALSE(current_time.has_value());
    CHECK(pending_times.empty());
  }

  // No completed ids, the pending id isn't the first, and this is a control
  // system, so do nothing
  {
    const LinkedMessageId<double> pending_time{2.0, {1.0}};
    pending_times.insert(pending_time);
    add_to_all_storage(pending_time, Destination::ControlSystem);
    set_current_time(make_not_null(&current_time),
                     make_not_null(&pending_times), completed_times,
                     all_storage, verbosity, name);
    CHECK_FALSE(current_time.has_value());
    CHECK(pending_times.contains(pending_time));
  }

  // No completed ids, the pending id isn't the first, and this is a
  // observation, so make it the current time
  {
    all_storage.clear();
    const LinkedMessageId<double> pending_time{2.0, {1.0}};
    pending_times.insert(pending_time);
    add_to_all_storage(pending_time, Destination::Observation);
    set_current_time(make_not_null(&current_time),
                     make_not_null(&pending_times), completed_times,
                     all_storage, verbosity, name);
    CHECK(current_time == std::optional{pending_time});
    CHECK(pending_times.empty());
  }

  // No completed ids and the pending id is the first, so make it the current
  // time
  {
    all_storage.clear();
    current_time.reset();
    const LinkedMessageId<double> pending_time{1.0, std::nullopt};
    pending_times.insert(pending_time);
    add_to_all_storage(pending_time, Destination::ControlSystem);
    set_current_time(make_not_null(&current_time),
                     make_not_null(&pending_times), completed_times,
                     all_storage, verbosity, name);
    CHECK(current_time == std::optional{pending_time});
    CHECK(pending_times.empty());
  }

  // One completed id, this is a control system, but the pending id isn't next
  // so do nothing
  {
    all_storage.clear();
    current_time.reset();
    pending_times.clear();
    completed_times.insert(LinkedMessageId<double>{1.0, std::nullopt});
    const LinkedMessageId<double> pending_time{3.0, {2.0}};
    pending_times.insert(pending_time);
    add_to_all_storage(pending_time, Destination::ControlSystem);
    set_current_time(make_not_null(&current_time),
                     make_not_null(&pending_times), completed_times,
                     all_storage, verbosity, name);
    CHECK_FALSE(current_time.has_value());
    CHECK(pending_times.contains(pending_time));
  }

  // One completed id and this is an observation, so use the next pending id
  {
    all_storage.clear();
    current_time.reset();
    pending_times.clear();
    completed_times.insert(LinkedMessageId<double>{1.0, std::nullopt});
    const LinkedMessageId<double> pending_time{3.0, {2.0}};
    pending_times.insert(pending_time);
    add_to_all_storage(pending_time, Destination::Observation);
    set_current_time(make_not_null(&current_time),
                     make_not_null(&pending_times), completed_times,
                     all_storage, verbosity, name);
    CHECK(current_time == std::optional{pending_time});
    CHECK(pending_times.empty());
  }

  // One completed id, and the pending id is next so make it the current time
  {
    all_storage.clear();
    current_time.reset();
    pending_times.clear();
    completed_times.insert(LinkedMessageId<double>{1.0, std::nullopt});
    const LinkedMessageId<double> pending_time{2.0, {1.0}};
    pending_times.insert(pending_time);
    add_to_all_storage(pending_time, Destination::ControlSystem);
    set_current_time(make_not_null(&current_time),
                     make_not_null(&pending_times), completed_times,
                     all_storage, verbosity, name);
    CHECK(current_time == std::optional{pending_time});
    CHECK_FALSE(pending_times.contains(pending_time));
  }
}

struct UpdateFoT {
  static void apply(
      const gsl::not_null<std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
          f_of_t_list,
      const std::string& f_of_t_name, const double update_time,
      DataVector update_deriv, const double new_expiration_time) {
    (*f_of_t_list)
        .at(f_of_t_name)
        ->update(update_time, std::move(update_deriv), new_expiration_time);
  }
};

struct MockHorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
  using time_tag = ::Tags::TimeAndPrevious<0>;

  using frame = ::Frame::Grid;

  // Don't need callbacks
  using horizon_find_callbacks = tmpl::list<>;
  using horizon_find_failure_callbacks = tmpl::list<>;

  using compute_tags_on_element = tmpl::list<>;

  static constexpr Destination destination = Destination::ControlSystem;

  static std::string name() { return "MockHorizonMetavars"; }
};

size_t call_count = 0;  // NOLINT
struct MockFindApparentHorizon {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const LinkedMessageId<double>& /*incoming_time*/,
                    const ElementId<3>& /*incoming_element_id*/,
                    const ::Mesh<3>& /*incoming_mesh*/,
                    Variables<ah::vars_to_interpolate_to_target<
                        3, MockHorizonMetavars::frame>>&& /*incoming_vars*/,
                    const std::optional<std::string>& /*dependency*/,
                    const bool vars_have_already_been_received = false) {
    ++call_count;
    CHECK(vars_have_already_been_received);
  }
};

template <typename Metavariables>
struct MockComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      ah::Component<Metavariables, MockHorizonMetavars>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<3>>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;

  using replace_these_simple_actions =
      tmpl::list<ah::FindApparentHorizon<MockHorizonMetavars>>;
  using with_these_simple_actions = tmpl::list<MockFindApparentHorizon>;
};

struct MockMetavariables {
  using component_list = tmpl::list<MockComponent<MockMetavariables>>;
};

void test_check_current_time() {
  (void)MockHorizonMetavars::destination;

  const auto domain_creator = domain::creators::Sphere(
      1.8, 2.2, domain::creators::Sphere::Excision{}, 1_st, 5_st, false,
      std::nullopt, std::vector<double>{},
      domain::CoordinateMaps::Distribution::Linear, ShellWedges::All,
      {std::make_unique<
          domain::creators::time_dependence::RotationAboutZAxis<3>>(0.0, 0.0,
                                                                    0.1, 0.0)});

  using component = MockComponent<MockMetavariables>;

  ActionTesting::MockRuntimeSystem<MockMetavariables> runner{
      {domain_creator.create_domain()},
      {domain_creator.functions_of_time({{"Rotation", 2.0}})}};

  ActionTesting::emplace_array_component<component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& cache = ActionTesting::cache<component>(runner, 0_st);

  const LinkedMessageId<double> incoming_time{100.0, {99.0}};
  const ElementId<3> incoming_element_id{1};
  const Mesh<3> incoming_mesh{2, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
  const std::optional<std::string> dependency{"FakeDependency"};
  LinkedMessageId<double> current_time{0.0, {std::nullopt}};

  // Current time is ready so no callback is registered
  CHECK(check_if_current_time_is_ready<MockHorizonMetavars>(
      current_time, cache, incoming_time, incoming_element_id, incoming_mesh,
      dependency));

  // Current time isn't ready so a callback should be registered
  current_time = LinkedMessageId<double>{2.5, {1.5}};
  CHECK_FALSE(check_if_current_time_is_ready<MockHorizonMetavars>(
      current_time, cache, incoming_time, incoming_element_id, incoming_mesh,
      dependency));

  // Mutate the function of time which will call (queue) the simple action
  Parallel::mutate<domain::Tags::FunctionsOfTime, UpdateFoT>(
      cache, "Rotation"s, 2.0, DataVector{0.0}, 3.0);
  REQUIRE(ActionTesting::number_of_queued_simple_actions<component>(runner,
                                                                    0) == 1);
  ActionTesting::invoke_queued_simple_action<component>(make_not_null(&runner),
                                                        0);
  CHECK(call_count == 1);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.CurrentTime",
                  "[ApparentHorizonFinder][Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  test_set_current_time();
  test_check_current_time();
}
}  // namespace ah
