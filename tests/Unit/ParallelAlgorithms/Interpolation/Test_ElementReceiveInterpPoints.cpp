// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/ElementInitInterpPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/ElementReceiveInterpPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetSendPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/LineSegment.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {

// Simple Variables tag for test.
namespace Tags {
struct TestSolution : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace Tags

template <typename Metavariables>
struct mock_element {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<intrp::Actions::ElementInitInterpPoints<
          intrp::Tags::InterpPointInfo<Metavariables>>>>>;
  using initial_databox = db::compute_databox_type<
      tmpl::list<intrp::Tags::InterpPointInfo<Metavariables>>>;
  using component_being_mocked =
      DgElementArray<Metavariables, phase_dependent_action_list>;
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  static_assert(
      tt::assert_conforms_to_v<InterpolationTargetTag,
                               intrp::protocols::InterpolationTargetTag>);
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::push_back<
      Parallel::get_const_global_cache_tags_from_actions<
          tmpl::list<typename InterpolationTargetTag::compute_target_points>>,
      domain::Tags::Domain<Metavariables::volume_dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolationTarget<
              Metavariables, InterpolationTargetTag>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Register,
          tmpl::list<
              intrp::Actions::InterpolationTargetSendTimeIndepPointsToElements<
                  InterpolationTargetTag>>>>;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
};

struct MockMetavariables {
  struct InterpolationTargetA
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target = tmpl::list<Tags::TestSolution>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        ::intrp::TargetPoints::LineSegment<InterpolationTargetA, 3,
                                           Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tmpl::list<>,
                                                     InterpolationTargetA>;
    template <typename Metavariables>
    using interpolating_component = mock_element<Metavariables>;
  };
  struct InterpolationTargetB
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target = tmpl::list<Tags::TestSolution>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        ::intrp::TargetPoints::LineSegment<InterpolationTargetB, 3,
                                           Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tmpl::list<>,
                                                     InterpolationTargetA>;
    template <typename Metavariables>
    using interpolating_component = mock_element<Metavariables>;
  };
  static constexpr size_t volume_dim = 3;
  using interpolation_target_tags =
      tmpl::list<InterpolationTargetA, InterpolationTargetB>;

  using component_list = tmpl::list<
      mock_interpolation_target<MockMetavariables, InterpolationTargetA>,
      mock_interpolation_target<MockMetavariables, InterpolationTargetB>,
      mock_element<MockMetavariables>>;
};

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.ElementReceivePoints",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();

  using metavars = MockMetavariables;
  using target_component_a =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;
  using target_component_b =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetB>;
  using elem_component = mock_element<metavars>;

  // Options
  const auto domain_creator = domain::creators::Sphere(
      0.9, 4.9, domain::creators::Sphere::Excision{}, 1_st, 5_st, false);
  intrp::OptionHolders::LineSegment<3> line_segment_opts_a(
      {{1.0, 1.0, 1.0}}, {{2.4, 2.4, 2.4}}, 15);
  intrp::OptionHolders::LineSegment<3> line_segment_opts_b(
      {{1.0, 1.0, 1.0}}, {{2.1, 2.1, 2.1}}, 12);
  tuples::TaggedTuple<intrp::Tags::LineSegment<metavars::InterpolationTargetA,
                                               metavars::volume_dim>,
                      intrp::Tags::LineSegment<metavars::InterpolationTargetB,
                                               metavars::volume_dim>,
                      domain::Tags::Domain<metavars::volume_dim>>
      tuple_of_opts{std::move(line_segment_opts_a),
                    std::move(line_segment_opts_b),
                    domain_creator.create_domain()};

  // Initialization
  ActionTesting::MockRuntimeSystem<metavars> runner{std::move(tuple_of_opts)};
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_component<target_component_a>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_component_a>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_component<target_component_b>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_component_b>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_component<elem_component>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<elem_component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Register);

  using point_info_type = tnsr::I<
      DataVector, metavars::volume_dim,
      typename metavars::InterpolationTargetA::compute_target_points::frame>;

  // element should contain an intrp::Tags::InterpPointInfo<Metavariables>
  // that contains a default-constructed point_info for InterpolationTargetA and
  // InterpolationTargetB. This tests ElementInitInterpPoints.
  const auto& init_point_infos =
      ActionTesting::get_databox_tag<elem_component,
                                     ::intrp::Tags::InterpPointInfo<metavars>>(
          runner, 0);
  CHECK(get<intrp::Vars::PointInfoTag<metavars::InterpolationTargetA,
                                      metavars::volume_dim>>(
            init_point_infos) == point_info_type{});
  CHECK(get<intrp::Vars::PointInfoTag<metavars::InterpolationTargetB,
                                      metavars::volume_dim>>(
            init_point_infos) == point_info_type{});

  // Now invoke the only Registration action (InterpolationTargetSendPoints).
  ActionTesting::next_action<target_component_a>(make_not_null(&runner), 0);
  ActionTesting::next_action<target_component_b>(make_not_null(&runner), 0);
  // ... and the only queued simple action (ElementReceiveInterpPoints).
  runner.invoke_queued_simple_action<elem_component>(0);
  runner.invoke_queued_simple_action<elem_component>(0);

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // After registration, element should contain an
  // intrp::Tags::InterpPointInfo<Metavariables> that has a
  // point_info for InterpolationTargetA/B containing the
  // expected point info.
  const auto expected_point_info_a = []() {
    const size_t n_pts = 15;
    tnsr::I<DataVector, 3, Frame::Inertial> points(n_pts);
    for (size_t d = 0; d < 3; ++d) {
      for (size_t i = 0; i < n_pts; ++i) {
        points.get(d)[i] = 1.0 + 0.1 * i;  // Worked out by hand.
      }
    }
    return points;
  }();
  const auto expected_point_info_b = []() {
    const size_t n_pts = 12;
    tnsr::I<DataVector, 3, Frame::Inertial> points(n_pts);
    for (size_t d = 0; d < 3; ++d) {
      for (size_t i = 0; i < n_pts; ++i) {
        points.get(d)[i] = 1.0 + 0.1 * i;  // Worked out by hand.
      }
    }
    return points;
  }();

  const auto& point_infos =
      ActionTesting::get_databox_tag<elem_component,
                                     ::intrp::Tags::InterpPointInfo<metavars>>(
          runner, 0);
  const auto& point_info_a =
      get<intrp::Vars::PointInfoTag<metavars::InterpolationTargetA,
                                    metavars::volume_dim>>(point_infos);
  const auto& point_info_b =
      get<intrp::Vars::PointInfoTag<metavars::InterpolationTargetB,
                                    metavars::volume_dim>>(point_infos);
  CHECK_ITERABLE_APPROX(point_info_a, expected_point_info_a);
  CHECK_ITERABLE_APPROX(point_info_b, expected_point_info_b);

  // Should be no queued simple actions on either component.
  CHECK(runner.is_simple_action_queue_empty<target_component_a>(0));
  CHECK(runner.is_simple_action_queue_empty<target_component_b>(0));
  CHECK(runner.is_simple_action_queue_empty<elem_component>(0));
}
}  // namespace
