// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/SphericalShells.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RotationAboutZAxis.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/Interpolation/InterpolateOnElementTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetVarsFromElement.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/InterpolateWithoutInterpComponent.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeVarsToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/LineSegment.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/Sphere.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename Metavariables>
struct mock_element {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
  using initial_databox = db::compute_databox_type<db::AddSimpleTags<>>;
};

template <typename Metavariables, typename ElemComponent>
struct initialize_elements_and_queue_simple_actions {
  template <typename Runner, typename TemporalId>
  void operator()(const DomainCreator<3>& domain_creator,
                  const Domain<3>& domain,
                  const std::vector<ElementId<3>>& element_ids,
                  const tnsr::I<DataVector, 3, Frame::Inertial>& target_points,
                  Runner& runner, const TemporalId& temporal_id) {
    using metavars = Metavariables;
    using elem_component = ElemComponent;
    // Emplace elements.
    for (const auto& element_id : element_ids) {
      ActionTesting::emplace_component<elem_component>(&runner, element_id);
      ActionTesting::next_action<elem_component>(make_not_null(&runner),
                                                 element_id);
    }
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

    // Create event.
    typename metavars::event event{};

    CHECK(event.needs_evolved_variables());

    // Run event on all elements.
    for (const auto& element_id : element_ids) {
      // 1. Get vars, mesh, and coords
      const auto& [vars, mesh, inertial_coords] =
          InterpolateOnElementTestHelpers::make_volume_data_and_mesh<
              ElemComponent, Metavariables::use_time_dependent_maps>(
              domain_creator, runner, domain, element_id, temporal_id);

      // 2. Make a box
      auto box = db::create<db::AddSimpleTags<
          Parallel::Tags::MetavariablesImpl<metavars>,
          typename metavars::InterpolationTargetA::temporal_id,
          intrp::Tags::PointInfo<typename metavars::InterpolationTargetA,
                                 tmpl::size_t<3>>,
          ::Events::Tags::ObserverMesh<metavars::volume_dim>,
          domain::Tags::Mesh<metavars::volume_dim>,
          domain::Tags::Coordinates<metavars::volume_dim, Frame::Inertial>,
          ::Tags::Variables<
              typename std::remove_reference_t<decltype(vars)>::tags_list>>>(
          metavars{}, temporal_id, target_points, mesh, mesh, inertial_coords,
          vars);

      // 3. Run the event.  This will invoke simple actions on
      // InterpolationTarget.
      auto obs_box = make_observation_box<
          typename metavars::event::compute_tags_for_observation_box>(
          make_not_null(&box));
      event.run(make_not_null(&obs_box),
                ActionTesting::cache<elem_component>(runner, element_id),
                element_id, std::add_pointer_t<elem_component>{}, {});
    }
  }
};

template <bool HaveComputeVarsToInterpolate, bool UseTimeDependentMaps>
struct MockMetavariables {
  static constexpr bool use_time_dependent_maps = UseTimeDependentMaps;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<3>>;
  using mutable_global_cache_tags =
      tmpl::conditional_t<use_time_dependent_maps,
                          tmpl::list<domain::Tags::FunctionsOfTimeInitialize>,
                          tmpl::list<>>;
  struct InterpolationTargetAWithComputeVarsToInterpolate
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using compute_items_on_target = tmpl::list<>;
    using vars_to_interpolate_to_target =
        tmpl::list<InterpolateOnElementTestHelpers::Tags::MultiplyByTwo>;
    using compute_vars_to_interpolate =
        InterpolateOnElementTestHelpers::ComputeMultiplyByTwo;
    // The following are not used in this test, but must be there to
    // conform to the protocol.
    using compute_target_points = ::intrp::TargetPoints::LineSegment<
        InterpolationTargetAWithComputeVarsToInterpolate, 3, Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<>, InterpolationTargetAWithComputeVarsToInterpolate>>;
  };
  struct InterpolationTargetAWithoutComputeVarsToInterpolate
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using compute_items_on_target = tmpl::list<>;
    using vars_to_interpolate_to_target =
        tmpl::list<InterpolateOnElementTestHelpers::Tags::TestSolution>;
    // The following are not used in this test, but must be there to
    // conform to the protocol.
    using compute_target_points = ::intrp::TargetPoints::Sphere<
        InterpolationTargetAWithoutComputeVarsToInterpolate, Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<>, InterpolationTargetAWithoutComputeVarsToInterpolate>>;
  };
  using InterpolationTargetA =
      tmpl::conditional_t<HaveComputeVarsToInterpolate,
                          InterpolationTargetAWithComputeVarsToInterpolate,
                          InterpolationTargetAWithoutComputeVarsToInterpolate>;
  static constexpr size_t volume_dim = 3;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;

  using component_list =
      tmpl::list<InterpolateOnElementTestHelpers::mock_interpolation_target<
                     MockMetavariables, InterpolationTargetA>,
                 mock_element<MockMetavariables>>;

  using event = intrp::Events::InterpolateWithoutInterpComponent<
      volume_dim, InterpolationTargetA,
      tmpl::list<InterpolateOnElementTestHelpers::Tags::TestSolution>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<Event, tmpl::list<event>>>;
  };
};

template <typename MockMetavariables, bool OffCenter = false>
void run_test() {
  using metavars = MockMetavariables;
  using elem_component = mock_element<metavars>;
  InterpolateOnElementTestHelpers::test_interpolate_on_element<
      metavars, elem_component, OffCenter>(
      initialize_elements_and_queue_simple_actions<metavars, elem_component>{});
}

// Test that every point of a Sphere target is received, with the correct
// value, when the sphere lies on (or within roundoff of) a radial element
// boundary inside a block.
namespace boundary_test {
using solution_tag = InterpolateOnElementTestHelpers::Tags::TestSolution;

// Interpolated values of the test solution by offset of the target point
struct ReceivedValues : db::SimpleTag {
  using type = std::unordered_map<size_t, double>;
};

struct RecordReceivedValues {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex, typename VarsSrc, typename TemporalId>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const VarsSrc& vars_src,
      const std::vector<BlockLogicalCoords<Metavariables::volume_dim>>&
      /*block_logical_coords*/,
      const std::vector<std::vector<size_t>>& global_offsets,
      const TemporalId& /*temporal_id*/) {
    db::mutate<ReceivedValues>(
        [&global_offsets,
         &vars_src](const gsl::not_null<std::unordered_map<size_t, double>*>
                        received_values) {
          for (size_t i = 0; i < global_offsets.size(); ++i) {
            const auto& values = get(get<solution_tag>(vars_src[i]));
            for (size_t j = 0; j < global_offsets[i].size(); ++j) {
              received_values->insert_or_assign(global_offsets[i][j],
                                                values[j]);
            }
          }
        },
        make_not_null(&box));
  }
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
  using simple_tags = tmpl::list<ReceivedValues>;
  using const_global_cache_tags =
      tmpl::list<intrp::Tags::Sphere<InterpolationTargetTag>,
                 intrp::Tags::Verbosity>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::InterpolationTargetVarsFromElement<
          InterpolationTargetTag>>;
  using with_these_simple_actions = tmpl::list<RecordReceivedValues>;
};

template <bool UseTimeDependentMaps>
struct Metavariables {
  static constexpr size_t volume_dim = 3;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<3>>;
  using mutable_global_cache_tags =
      tmpl::conditional_t<UseTimeDependentMaps,
                          tmpl::list<domain::Tags::FunctionsOfTimeInitialize>,
                          tmpl::list<>>;
  struct SphereTarget
      : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using compute_items_on_target = tmpl::list<>;
    using vars_to_interpolate_to_target = tmpl::list<solution_tag>;
    using compute_target_points =
        ::intrp::TargetPoints::Sphere<SphereTarget, Frame::Inertial>;
    using post_interpolation_callbacks =
        tmpl::list<intrp::callbacks::ObserveTimeSeriesOnSurface<tmpl::list<>,
                                                                SphereTarget>>;
  };
  using interpolation_target_tags = tmpl::list<SphereTarget>;
  using component_list =
      tmpl::list<mock_interpolation_target<Metavariables, SphereTarget>,
                 mock_element<Metavariables>>;
  using event = intrp::Events::InterpolateWithoutInterpComponent<
      volume_dim, SphereTarget, tmpl::list<solution_tag>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<Event, tmpl::list<event>>>;
  };
};

// Runs the event on all elements of the domain and checks that every point of
// the sphere target is received with the value of the linear test solution
// up to `relative_tolerance`. The radii are on the element
// boundary at `boundary_radius` and within roundoff of it on either side, so
// points just inside the lower element must not be dropped. If `center` is
// not the center of the domain, the radii are not on element boundaries but
// the sphere crosses several elements.
template <bool UseTimeDependentMaps>
void test_sphere_on_radial_element_boundary(
    const DomainCreator<3>& domain_creator, const double boundary_radius,
    const double relative_tolerance,
    const std::array<double, 3>& center = {{0.0, 0.0, 0.0}}) {
  using metavars = Metavariables<UseTimeDependentMaps>;
  using target_tag = typename metavars::SphereTarget;
  using target_component = mock_interpolation_target<metavars, target_tag>;
  using elem_component = mock_element<metavars>;

  const std::vector<ElementId<3>> element_ids =
      initial_element_ids(domain_creator.initial_refinement_levels());

  const std::vector<double> radii{boundary_radius * (1.0 - 1.0e-14),
                                  boundary_radius,
                                  boundary_radius * (1.0 + 1.0e-14)};
  const size_t l_max = 4;
  const size_t num_theta = l_max + 1;
  const size_t num_phi = 2 * l_max + 1;
  const size_t num_points = radii.size() * num_theta * num_phi;
  // The angular points include the equator and phi = 0, e.g. the center of the
  // +x wedge's faces.
  tnsr::I<DataVector, 3, Frame::Inertial> target_points(num_points);
  for (size_t r = 0, s = 0; r < radii.size(); ++r) {
    for (size_t j = 0; j < num_phi; ++j) {
      for (size_t i = 0; i < num_theta; ++i, ++s) {
        const double theta = std::numbers::pi * (static_cast<double>(i) + 0.5) /
                             static_cast<double>(num_theta);
        const double phi = 2.0 * std::numbers::pi * static_cast<double>(j) /
                           static_cast<double>(num_phi);
        get<0>(target_points)[s] = center[0] + radii[r] * sin(theta) * cos(phi);
        get<1>(target_points)[s] = center[1] + radii[r] * sin(theta) * sin(phi);
        get<2>(target_points)[s] = center[2] + radii[r] * cos(theta);
      }
    }
  }

  tuples::TaggedTuple<domain::Tags::Domain<3>, intrp::Tags::Sphere<target_tag>,
                      intrp::Tags::Verbosity>
      init_tuple{domain_creator.create_domain(),
                 intrp::OptionHolders::Sphere{l_max, center, radii,
                                              ylm::AngularOrdering::Cce},
                 ::Verbosity::Silent};
  auto runner = [&domain_creator, &init_tuple]() {
    if constexpr (UseTimeDependentMaps) {
      return ActionTesting::MockRuntimeSystem<metavars>(
          std::move(init_tuple), domain_creator.functions_of_time());
    } else {
      (void)domain_creator;
      return ActionTesting::MockRuntimeSystem<metavars>(std::move(init_tuple));
    }
  }();
  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_component_and_initialize<target_component>(
      &runner, 0, {std::unordered_map<size_t, double>{}});
  for (const auto& element_id : element_ids) {
    ActionTesting::emplace_component<elem_component>(&runner, element_id);
    ActionTesting::next_action<elem_component>(make_not_null(&runner),
                                               element_id);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const Slab slab(0.0, 10.0);
  const TimeStepId temporal_id(true, 0, Time(slab, Rational(73, 100)));
  const typename metavars::event event{};
  for (const auto& element_id : element_ids) {
    const auto& block =
        Parallel::get<domain::Tags::Domain<3>>(
            ActionTesting::cache<elem_component>(runner, element_id))
            .blocks()[element_id.block_id()];
    const Mesh<3> mesh = domain::create_initial_mesh(
        domain_creator.initial_extents(), block, element_id,
        Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
    const auto inertial_coords = [&]() {
      if constexpr (UseTimeDependentMaps) {
        const ElementMap<3, Frame::Grid> map_logical_to_grid{
            element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
        return block.moving_mesh_grid_to_inertial_map()(
            map_logical_to_grid(logical_coordinates(mesh)),
            temporal_id.substep_time(),
            get<domain::Tags::FunctionsOfTime>(
                ActionTesting::cache<elem_component>(runner, element_id)));
      } else {
        const ElementMap<3, Frame::Inertial> map{
            element_id, block.stationary_map().get_clone()};
        return map(logical_coordinates(mesh));
      }
    }();
    Variables<tmpl::list<solution_tag>> vars(mesh.number_of_grid_points());
    InterpolateOnElementTestHelpers::fill_variables<solution_tag>(
        make_not_null(&vars), inertial_coords);

    auto box = db::create<db::AddSimpleTags<
        Parallel::Tags::MetavariablesImpl<metavars>,
        typename target_tag::temporal_id,
        intrp::Tags::PointInfo<target_tag, tmpl::size_t<3>>,
        ::Events::Tags::ObserverMesh<3>, domain::Tags::Mesh<3>,
        domain::Tags::Coordinates<3, Frame::Inertial>,
        ::Tags::Variables<tmpl::list<solution_tag>>>>(metavars{}, temporal_id,
                                                      target_points, mesh, mesh,
                                                      inertial_coords, vars);
    auto obs_box = make_observation_box<
        typename metavars::event::compute_tags_for_observation_box>(
        make_not_null(&box));
    event.run(make_not_null(&obs_box),
              ActionTesting::cache<elem_component>(runner, element_id),
              element_id, std::add_pointer_t<elem_component>{}, {});
  }
  while (not ActionTesting::is_simple_action_queue_empty<target_component>(
      runner, 0)) {
    runner.template invoke_queued_simple_action<target_component>(0);
  }

  const auto& received_values =
      ActionTesting::get_databox_tag<target_component, ReceivedValues>(runner,
                                                                       0);
  Variables<tmpl::list<solution_tag>> expected_vars(num_points);
  InterpolateOnElementTestHelpers::fill_variables<solution_tag>(
      make_not_null(&expected_vars), target_points);
  const auto& expected_values = get(get<solution_tag>(expected_vars));
  for (size_t i = 0; i < num_points; ++i) {
    CAPTURE(i);
    CAPTURE(get<0>(target_points)[i]);
    CAPTURE(get<1>(target_points)[i]);
    CAPTURE(get<2>(target_points)[i]);
    REQUIRE(received_values.contains(i));
    CHECK(received_values.at(i) == Approx::custom()
                                       .epsilon(relative_tolerance)
                                       .scale(1.0)(expected_values[i]));
  }
}

void test_sphere_on_radial_element_boundary() {
  {
    INFO("Wedges with Legendre bases");
    test_sphere_on_radial_element_boundary<false>(
        domain::creators::Sphere{0.9, 2.9, domain::creators::Sphere::Excision{},
                                 std::array<size_t, 3>{{0, 0, 2}}, 6_st, false},
        2.4, 1.0e-2);
  }
  {
    INFO("Spherical-harmonic shells");
    test_sphere_on_radial_element_boundary<false>(
        domain::creators::SphericalShells{0.9, 2.9, 2, 6, 7}, 2.4, 1.0e-12);
  }
  {
    INFO("Rotating spherical-harmonic shells");
    test_sphere_on_radial_element_boundary<true>(
        domain::creators::SphericalShells{
            0.9,
            2.9,
            2,
            6,
            7,
            {},
            domain::CoordinateMaps::Distribution::Linear,
            std::make_unique<
                domain::creators::time_dependence::RotationAboutZAxis<3>>(
                0.0, 0.0, 0.1, 0.0)},
        2.4, 1.0e-12);
  }
  {
    INFO("Spherical-harmonic shells, off-center sphere");
    test_sphere_on_radial_element_boundary<false>(
        domain::creators::SphericalShells{0.9, 2.9, 2, 6, 7}, 2.0, 1.0e-12,
        {{0.3, -0.2, 0.1}});
  }
  {
    INFO("Spherical-harmonic shells, sphere on a block boundary");
    test_sphere_on_radial_element_boundary<false>(
        domain::creators::SphericalShells{0.9, 2.9, 0, 6, 7, {2.4}}, 2.4,
        1.0e-12);
  }
  {
    INFO("Filled ball and spherical-harmonic shell");
    test_sphere_on_radial_element_boundary<false>(
        domain::creators::SphericalShells{0.0, 2.9, 0, 6, 7, {1.4}}, 1.4,
        1.0e-12);
  }
}

}  // namespace boundary_test

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolator.InterpolateEventNoInterpolator",
    "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  run_test<MockMetavariables<false, false>>();
  run_test<MockMetavariables<true, false>>();
  run_test<MockMetavariables<false, true>>();
  run_test<MockMetavariables<true, true>>();

  // Off-center test
  run_test<MockMetavariables<false, false>, true>();

  boundary_test::test_sphere_on_radial_element_boundary();
}
}  // namespace
