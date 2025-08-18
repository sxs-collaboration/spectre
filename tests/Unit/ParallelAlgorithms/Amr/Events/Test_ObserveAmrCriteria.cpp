// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/ParallelAlgorithms/Events/ObserveFields.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
#include "ParallelAlgorithms/Amr/Events/ObserveAmrCriteria.hpp"
#include "ParallelAlgorithms/Amr/Policies/Isotropy.hpp"
#include "ParallelAlgorithms/Amr/Policies/Limits.hpp"
#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"
#include "ParallelAlgorithms/Amr/Policies/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<
      TestHelpers::dg::Events::ObserveFields::ElementComponent<Metavariables>,
      TestHelpers::dg::Events::ObserveFields::MockObserverComponent<
          Metavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            amr::Criterion,
            tmpl::list<
                amr::Criteria::DriveToTarget<1, amr::Criteria::Type::h>,
                amr::Criteria::DriveToTarget<1, amr::Criteria::Type::p>>>,
        tmpl::pair<Event,
                   tmpl::list<amr::Events::ObserveAmrCriteria<Metavariables>>>>;
  };
};

void test() {
  std::vector<std::unique_ptr<amr::Criterion>> criteria;
  criteria.emplace_back(
      std::make_unique<amr::Criteria::DriveToTarget<1, amr::Criteria::Type::h>>(
          std::array{1_st}, std::array{amr::Flag::DoNothing}));
  criteria.emplace_back(
      std::make_unique<amr::Criteria::DriveToTarget<1, amr::Criteria::Type::p>>(
          std::array{3_st}, std::array{amr::Flag::DoNothing}));
  criteria.emplace_back(
      std::make_unique<amr::Criteria::DriveToTarget<1, amr::Criteria::Type::p>>(
          std::array{5_st}, std::array{amr::Flag::DoNothing}));
  criteria.emplace_back(
      std::make_unique<amr::Criteria::DriveToTarget<1, amr::Criteria::Type::h>>(
          std::array{0_st}, std::array{amr::Flag::DoNothing}));
  criteria.emplace_back(
      std::make_unique<amr::Criteria::DriveToTarget<1, amr::Criteria::Type::h>>(
          std::array{2_st}, std::array{amr::Flag::DoNothing}));
  const size_t number_of_criteria = criteria.size();
  const std::vector<double> expected_values{0.0, -1.0, 1.0, -2.0, 2.0};
  register_factory_classes_with_charm<Metavariables>();
  using element_component =
      TestHelpers::dg::Events::ObserveFields::ElementComponent<Metavariables>;
  using observer_component =
      TestHelpers::dg::Events::ObserveFields::MockObserverComponent<
          Metavariables>;
  element_component* const element_component_p = nullptr;

  const ElementId<1> element_id(0, make_array<1>(SegmentId(1, 0)));
  const Mesh<1> mesh{4_st, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      element_id);
  ActionTesting::emplace_group_component<observer_component>(&runner);
  auto& cache = ActionTesting::cache<element_component>(runner, element_id);

  const auto event =
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
          "ObserveAmrCriteria:\n"
          "  SubfileName: amr_criteria\n"
          "  CoordinatesFloatingPointType: Double\n"
          "  FloatingPointType: Float");
  const double time = 3.0;
  Domain domain(make_vector(
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<1>{})));
  domain.inject_time_dependent_map_for_block(
      0, domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
             domain::CoordinateMaps::TimeDependent::Translation<1>(
                 "translation")));

  domain::FunctionsOfTimeMap functions_of_time{};
  functions_of_time.emplace(
      "translation",
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<1>>(
          1.0, std::array{DataVector(1, 2.0), DataVector(1, 5.0)}, 4.0));
  const double expected_offset = 2.0 + (time - 1.0) * 5.0;

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                        Tags::Time, domain::Tags::FunctionsOfTime,
                        domain::Tags::Domain<1>, domain::Tags::Mesh<1>,
                        amr::Criteria::Tags::Criteria, amr::Tags::Policies>>(
      Metavariables{}, time, std::move(functions_of_time), std::move(domain),
      mesh, std::move(criteria),
      amr::Policies{amr::Isotropy::Anisotropic, amr::Limits{}, true, true});

  const double observation_value = 1.23;

  auto observation_box =
      make_observation_box<typename amr::Events::ObserveAmrCriteria<
          Metavariables>::compute_tags_for_observation_box>(
          make_not_null(&box));

  event->run(make_not_null(&observation_box), cache, element_id,
             element_component_p, {"value_name", observation_value});

  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results =
      TestHelpers::dg::Events::ObserveFields::MockContributeVolumeData::results;
  CHECK(results.observation_id.value() == observation_value);
  CHECK(results.observation_id.observation_key().tag() == "/amr_criteria.vol");
  CHECK(results.subfile_name == "/amr_criteria");
  CHECK(results.array_component_id ==
        Parallel::make_array_component_id<element_component>(element_id));
  CHECK(results.received_volume_data.element_name == get_output(element_id));
  CHECK(results.received_volume_data.extents == std::vector<size_t>(1, 2));
  const auto& components = results.received_volume_data.tensor_components;
  REQUIRE(components.size() == 1 + number_of_criteria);
  for (const auto& component : components) {
    std::visit([](const auto& data) { CHECK(data.size() == 2); },
               component.data);
  }
  CHECK(components[0].name == "InertialCoordinates_x");
  std::visit(
      [&](const auto& data) {
        for (size_t i = 0; i < data.size(); ++i) {
          CHECK(data[i] ==
                (i % 2 < 1 ? -1.0 + expected_offset : expected_offset));
        }
      },
      components[0].data);
  for (size_t i = 0; i < number_of_criteria; ++i) {
    CHECK(components[i + 1].name == "DriveToTarget_x");
    std::visit(
        [&](const auto& data) {
          for (size_t j = 0; j < data.size(); ++j) {
            CHECK(data[j] == expected_values[i]);
          }
        },
        components[i + 1].data);
  }
}

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Amr.Events.ObserveAmrCriteria",
                  "[Unit][ParallelAlgorithms]") {
  test();
}
}  // namespace
