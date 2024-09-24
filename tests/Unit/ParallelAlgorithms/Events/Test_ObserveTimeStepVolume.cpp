// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
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
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStepVolume.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct BlockLogical;
struct Grid;
struct Inertial;
}  // namespace Frame

namespace {
namespace LocalTags {
struct FunctionsOfTime : domain::Tags::FunctionsOfTime, db::SimpleTag {
  using type = domain::FunctionsOfTimeMap;
};
}  // namespace LocalTags

template <size_t VolumeDim>
struct System {
  static constexpr size_t volume_dim = VolumeDim;
};

template <size_t VolumeDim>
struct Metavariables {
  using system = System<VolumeDim>;
  using component_list = tmpl::list<
      TestHelpers::dg::Events::ObserveFields::ElementComponent<Metavariables>,
      TestHelpers::dg::Events::ObserveFields::MockObserverComponent<
          Metavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event,
                   tmpl::list<dg::Events::ObserveTimeStepVolume<VolumeDim>>>,
        tmpl::pair<Trigger, tmpl::list<Triggers::Always>>>;
  };
};

template <size_t VolumeDim>
void test() {
  using metavars = Metavariables<VolumeDim>;
  using element_component =
      TestHelpers::dg::Events::ObserveFields::ElementComponent<metavars>;
  using observer_component =
      TestHelpers::dg::Events::ObserveFields::MockObserverComponent<metavars>;
  element_component* const element_component_p = nullptr;

  const ElementId<VolumeDim> element_id(0,
                                        make_array<VolumeDim>(SegmentId(1, 0)));

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      element_id);
  ActionTesting::emplace_group_component<observer_component>(&runner);
  auto& cache = ActionTesting::cache<element_component>(runner, element_id);

  const auto events_and_triggers =
      TestHelpers::test_creation<EventsAndTriggers, metavars>(
          "- Trigger: Always\n"
          "  Events:\n"
          "    - ObserveTimeStepVolume:\n"
          "        SubfileName: time_step\n"
          "        CoordinatesFloatingPointType: Double\n"
          "        FloatingPointType: Float");

  const double time = 3.0;
  const auto time_step = Slab(4.0, 6.0).duration() / 4;

  Domain domain(make_vector(
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<VolumeDim>{})));
  domain.inject_time_dependent_map_for_block(
      0, domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
             domain::CoordinateMaps::TimeDependent::Translation<VolumeDim>(
                 "translation")));

  domain::FunctionsOfTimeMap functions_of_time{};
  functions_of_time.emplace(
      "translation",
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<1>>(
          1.0,
          std::array{DataVector(VolumeDim, 2.0), DataVector(VolumeDim, 5.0)},
          4.0));
  const double expected_offset = 2.0 + (time - 1.0) * 5.0;

  auto box = db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<metavars>, Tags::Time,
      LocalTags::FunctionsOfTime, domain::Tags::Domain<VolumeDim>,
      Tags::TimeStep,
      domain::Tags::MinimumGridSpacing<VolumeDim, Frame::Inertial>>>(
      metavars{}, time, std::move(functions_of_time), std::move(domain),
      time_step, 0.23);

  const double observation_value = 1.23;

  events_and_triggers.run_events(make_not_null(&box), cache, element_id,
                                 element_component_p,
                                 {"value_name", observation_value});

  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results =
      TestHelpers::dg::Events::ObserveFields::MockContributeVolumeData::results;
  CHECK(results.observation_id.value() == observation_value);
  CHECK(results.observation_id.observation_key().tag() == "/time_step.vol");
  CHECK(results.subfile_name == "/time_step");
  CHECK(results.array_component_id ==
        Parallel::make_array_component_id<element_component>(element_id));
  CHECK(results.received_volume_data.element_name == get_output(element_id));
  CHECK(results.received_volume_data.extents ==
        std::vector<size_t>(VolumeDim, 2));
  const auto& components = results.received_volume_data.tensor_components;
  REQUIRE(components.size() == VolumeDim + 3);
  for (const auto& component : components) {
    std::visit(
        [](const auto& data) { CHECK(data.size() == two_to_the(VolumeDim)); },
        component.data);
  }
  {
    // Element with SegmentId(1, 0)
    const double expected_lower = -1.0 + expected_offset;
    const double expected_upper = expected_offset;
    CHECK(components[0].name == "InertialCoordinates_x");
    std::visit(
        [&](const auto& data) {
          for (size_t i = 0; i < data.size(); ++i) {
            CHECK(data[i] == (i % 2 < 1 ? expected_lower : expected_upper));
          }
        },
        components[0].data);
    if constexpr (VolumeDim >= 2) {
      CHECK(components[1].name == "InertialCoordinates_y");
      std::visit(
          [&](const auto& data) {
            for (size_t i = 0; i < data.size(); ++i) {
              CHECK(data[i] == (i % 4 < 2 ? expected_lower : expected_upper));
            }
          },
          components[1].data);
    }
    if constexpr (VolumeDim >= 3) {
      CHECK(components[2].name == "InertialCoordinates_z");
      std::visit(
          [&](const auto& data) {
            for (size_t i = 0; i < data.size(); ++i) {
              CHECK(data[i] == (i % 8 < 4 ? expected_lower : expected_upper));
            }
          },
          components[2].data);
    }
  }
  CHECK(components[VolumeDim].name == "Time step");
  std::visit(
      [&](const auto& data) {
        for (size_t i = 0; i < data.size(); ++i) {
          CHECK(data[i] == time_step.value());
        }
      },
      components[VolumeDim].data);
  CHECK(components[VolumeDim + 1].name == "Slab fraction");
  std::visit(
      [&](const auto& data) {
        for (size_t i = 0; i < data.size(); ++i) {
          CHECK(data[i] == time_step.fraction().value());
        }
      },
      components[VolumeDim + 1].data);
  CHECK(components[VolumeDim + 2].name == "Minimum grid spacing");
  std::visit(
      [&](const auto& data) {
        for (size_t i = 0; i < data.size(); ++i) {
          CHECK(data[i] == 0.23f);
        }
      },
      components[VolumeDim + 2].data);
}

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Events.ObserveTimeStepVolume",
                  "[Unit][ParallelAlgorithms]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
