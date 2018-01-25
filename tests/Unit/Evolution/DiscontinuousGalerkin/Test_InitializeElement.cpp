// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <deque>
#include <memory>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/DomainCreators/Brick.hpp"
#include "Domain/DomainCreators/Interval.hpp"
#include "Domain/DomainCreators/Rectangle.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/InitializeElement.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct Var : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString label = "Var";
};

struct SystemAnalyticSolution {
  template <size_t Dim>
  Variables<tmpl::list<Var>> evolution_variables(
      const tnsr::I<DataVector, Dim>& x, double t) const noexcept {
    Variables<tmpl::list<Var>> variables(x.begin()->size());
    get(get<Var>(variables)) = x.get(0) + t;
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Var>(variables)) += x.get(d) + t;
    }
    return variables;
  }

  template <size_t Dim>
  Variables<tmpl::list<Tags::dt<Var>>> dt_evolution_variables(
      const tnsr::I<DataVector, Dim>& x, double t) const noexcept {
    Variables<tmpl::list<Tags::dt<Var>>> variables(x.begin()->size());
    get(get<Tags::dt<Var>>(variables)) = 2.0 * x.get(0) + t;
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Tags::dt<Var>>(variables)) += 2.0 * x.get(d) + t;
    }
    return variables;
  }
};

struct System {
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
  using gradients_tags = tmpl::list<Var>;
};

template <size_t Dim>
struct Metavariables;

template <size_t Dim>
using component = ActionTesting::MockArrayComponent<
    Metavariables<Dim>, ElementIndex<Dim>,
    tmpl::list<CacheTags::TimeStepper,
               CacheTags::AnalyticSolution<SystemAnalyticSolution>>,
    tmpl::list<dg::Actions::InitializeElement<Dim>>>;

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<component<Dim>>;
  using system = System;
};

template <size_t Dim, typename DomainCreatorType>
void test_initialize_element(const ElementId<Dim>& element_id,
                             const DomainCreatorType& domain_creator) {
  ActionTesting::ActionRunner<Metavariables<Dim>> runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(4, false),
       SystemAnalyticSolution{}}};

  const Slab slab = Slab::with_duration_from_start(0.3, 0.01);

  const auto domain = domain_creator.create_domain();

  db::DataBox<tmpl::list<>> empty_box{};
  auto box = std::get<0>(
      runner
          .template apply<component<Dim>, dg::Actions::InitializeElement<Dim>>(
              empty_box, element_id, domain_creator.initial_extents(),
              domain_creator.create_domain(), slab.start(), slab.duration()));
  CHECK(db::get<Tags::TimeId>(box) == [&slab]() {
    TimeId time_id{};
    time_id.time = slab.start();
    return time_id;
  }());
  CHECK(db::get<Tags::Time>(box) == slab.start());
  CHECK(db::get<Tags::TimeStep>(box) == slab.duration());

  const auto& my_block = domain.blocks()[element_id.block_id()];
  ElementMap<Dim, Frame::Inertial> map{element_id,
                                       my_block.coordinate_map().get_clone()};
  Element<Dim> element = create_initial_element(element_id, my_block);
  Index<Dim> extents{domain_creator.initial_extents()[element_id.block_id()]};
  const auto num_grid_points = extents.product();
  auto logical_coords = logical_coordinates(extents);
  auto inertial_coords = map(logical_coords);
  CHECK(db::get<Tags::LogicalCoordinates<Dim>>(box) == logical_coords);
  CHECK(db::get<Tags::Extents<Dim>>(box) == extents);
  CHECK(db::get<Tags::Element<Dim>>(box) == element);
  // Can't test ElementMap directly, only via inverse jacobian and grid
  // coordinates. We can check that we can retrieve it.
  (void)db::get<Tags::ElementMap<Dim>>(box);
  CHECK(db::get<Var>(box) == ([&inertial_coords, &slab]() {
          double time = slab.start().value();
          Scalar<DataVector> var{inertial_coords.get(0) + time};
          for (size_t d = 1; d < Dim; ++d) {
            get(var) += inertial_coords.get(d) + time;
          }
          return var;
        }()));
  {
    const auto& history = db::get<Tags::HistoryEvolvedVariables<
        typename System::variables_tag,
        db::add_tag_prefix<Tags::dt, typename System::variables_tag>>>(box);
    TimeSteppers::AdamsBashforthN stepper(4, false);
    CHECK(history.size() == stepper.number_of_past_steps());
    const SystemAnalyticSolution solution{};
    Time past_t{slab.start()};
    TimeDelta past_dt{slab.duration()};
    for (size_t i = stepper.number_of_past_steps(); i > 0; --i) {
      const auto entry = history.begin() + static_cast<ssize_t>(i - 1);
      past_dt = past_dt.with_slab(past_dt.slab().advance_towards(-past_dt));
      past_t -= past_dt;

      CHECK(*entry == past_t);
      CHECK(entry.value() ==
            solution.evolution_variables(inertial_coords, past_t.value()));
      CHECK(entry.derivative() ==
            solution.dt_evolution_variables(inertial_coords, past_t.value()));
    }
  }
  CHECK((db::get<Tags::Coordinates<Tags::ElementMap<Dim>,
                                   Tags::LogicalCoordinates<Dim>>>(box)) ==
        inertial_coords);
  CHECK((db::get<Tags::InverseJacobian<Tags::ElementMap<Dim>,
                                       Tags::LogicalCoordinates<Dim>>>(box)) ==
        map.inv_jacobian(logical_coords));
  CHECK((db::get<
            Tags::deriv<typename System::variables_tag::tags_list,
                        typename System::gradients_tags,
                        Tags::InverseJacobian<Tags::ElementMap<Dim>,
                                              Tags::LogicalCoordinates<Dim>>>>(
            box)) ==
        partial_derivatives<tmpl::list<Var>>(
            SystemAnalyticSolution{}.evolution_variables(inertial_coords, 0.),
            extents, map.inv_jacobian(logical_coords)));
  CHECK((db::get<db::add_tag_prefix<Tags::dt, typename System::variables_tag>>(
            box)) ==
        Variables<tmpl::list<Tags::dt<Var>>>(extents.product(), 0.0));
  CHECK(db::get<Tags::UnnormalizedFaceNormal<Dim>>(box) ==
        Tags::UnnormalizedFaceNormal<Dim>::function(extents, map));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.dG.InitializeElement",
                  "[Unit][Evolution][Actions]") {
  test_initialize_element(ElementId<1>{0, {{SegmentId{2, 1}}}},
                          DomainCreators::Interval<Frame::Inertial>{
                              {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}});

  test_initialize_element(
      ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{3, 2}}}},
      DomainCreators::Rectangle<Frame::Inertial>{
          {{-0.5, -0.75}}, {{1.5, 2.4}}, {{false, false}}, {{2, 3}}, {{4, 5}}});

  test_initialize_element(
      ElementId<3>{0, {{SegmentId{2, 1}, SegmentId{3, 2}, SegmentId{1, 0}}}},
      DomainCreators::Brick<Frame::Inertial>{{{-0.5, -0.75, -1.2}},
                                             {{1.5, 2.4, 1.2}},
                                             {{false, false, true}},
                                             {{2, 3, 1}},
                                             {{4, 5, 3}}});
}

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.cpp"
