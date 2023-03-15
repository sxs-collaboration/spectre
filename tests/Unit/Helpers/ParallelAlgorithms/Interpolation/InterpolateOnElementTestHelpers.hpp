// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetVarsFromElement.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/PointInfoTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeVarsToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// Holds code that is shared between multiple tests. Currently used by
/// - Test_InterpolateWithoutInterpolatorComponent.
/// - Test_SendNextTimeToCce.
namespace InterpolateOnElementTestHelpers {

namespace Tags {
// Simple Variables tag for test.
struct TestSolution : db::SimpleTag {
  using type = Scalar<DataVector>;
};
// Tag holding inertial-frame target points for test.
struct TestTargetPoints : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};
struct MultiplyByTwo : db::SimpleTag {
  using type = Scalar<DataVector>;
};
// Compute tag for test.
struct MultiplyByTwoCompute : MultiplyByTwo, db::ComputeTag {
  static void function(const gsl::not_null<Scalar<DataVector>*> result,
                       const Scalar<DataVector>& x) {
    get<>(*result) = get<>(x) * 2.0;
  }
  using return_type = Scalar<DataVector>;
  using argument_tags = tmpl::list<TestSolution>;
  using base = MultiplyByTwo;
};
}  // namespace Tags

// compute_vars_to_interpolate for test.
struct ComputeMultiplyByTwo
    : tt::ConformsTo<intrp::protocols::ComputeVarsToInterpolate> {
  // Although we know the explicit types for SrcTagList and DestTagList,
  // we keep SrcTagList and DestTagList as template parameters so
  // that Protocols work properly.
  template <typename SrcTagList, typename DestTagList>
  static void apply(const gsl::not_null<Variables<DestTagList>*> target_vars,
                    const Variables<SrcTagList>& src_vars,
                    const Mesh<3>& /*mesh*/) {
    static_assert(std::is_same_v<SrcTagList, tmpl::list<Tags::TestSolution>>,
                  "SrcTagList must be only TestSolution");
    static_assert(std::is_same_v<DestTagList, tmpl::list<Tags::MultiplyByTwo>>,
                  "DestTagList must be only MultiplyByTwo");
    const auto& src = get<Tags::TestSolution>(src_vars);
    auto& dest = get<Tags::MultiplyByTwo>(*target_vars);
    get(dest) = get(src) * 2.0;
  }
  using allowed_src_tags = tmpl::list<Tags::TestSolution>;
  using required_src_tags = tmpl::list<Tags::TestSolution>;
  // The following are required to be templated on Frame by the protocol.
  template <typename Frame>
  using allowed_dest_tags = tmpl::list<Tags::MultiplyByTwo>;
  template <typename Frame>
  using required_dest_tags = tmpl::list<Tags::MultiplyByTwo>;
};

template <typename TagName, typename VarsTagList>
void fill_variables(const gsl::not_null<Variables<VarsTagList>*> vars,
                    const tnsr::I<DataVector, 3, Frame::Inertial>& coords) {
  // Some analytic solution used to fill the volume data and
  // to test the interpolated data.
  auto& solution = get<TagName>(*vars);
  get(solution) =
      (2.0 * get<0>(coords) + 3.0 * get<1>(coords) + 5.0 * get<2>(coords));

  if constexpr (std::is_same_v<TagName, Tags::MultiplyByTwo>) {
    get(solution) *= 2.0;
  } else if constexpr (not std::is_same_v<TagName, Tags::TestSolution>) {
    ERROR("Do not understand the given TagName");
  }
}

template <typename InterpolationTargetTag>
struct MockInterpolationTargetVarsFromElement {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex, typename TemporalId,
      Requires<tmpl::list_contains_v<DbTags, Tags::TestTargetPoints>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const std::vector<Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>&
          vars_src,
      const std::vector<std::optional<
          IdPair<domain::BlockId, tnsr::I<double, Metavariables::volume_dim,
                                          typename ::Frame::BlockLogical>>>>&
      /*block_logical_coords*/,
      const std::vector<std::vector<size_t>>& global_offsets,
      const TemporalId& /*temporal_id*/) {
    CHECK(global_offsets.size() == vars_src.size());
    // global_offsets and vars_src always have a size of 1 for calls
    // directly from the elements; the outer vector is used only by
    // the Interpolator parallel component.
    CHECK(global_offsets.size() == 1);

    // Here we have received only some of the points.
    const size_t num_pts_received = global_offsets[0].size();

    // Create a new target_points containing only the ones we have received.
    const auto& all_target_points = db::get<Tags::TestTargetPoints>(box);
    tnsr::I<DataVector, 3, Frame::Inertial> target_points(num_pts_received);
    for (size_t i = 0; i < num_pts_received; ++i) {
      for (size_t d = 0; d < 3; ++d) {
        target_points.get(d)[i] =
            all_target_points.get(d)[global_offsets[0][i]];
      }
    }

    static_assert(
        tmpl::count<typename InterpolationTargetTag::
                        vars_to_interpolate_to_target>::value == 1,
        "For this test, we assume only a single interpolated variable");
    using solution_tag = tmpl::front<
        typename InterpolationTargetTag::vars_to_interpolate_to_target>;
    const auto& test_solution = get<solution_tag>(vars_src[0]);

    // Expected solution
    Variables<tmpl::list<solution_tag>> expected_vars(vars_src[0].size());
    fill_variables<solution_tag>(make_not_null(&expected_vars), target_points);
    const auto& expected_solution = get<solution_tag>(expected_vars);

    // We don't have that many points, so interpolation is good for
    // only a few digits.
    Approx custom_approx = Approx::custom().epsilon(1.e-5).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(test_solution, expected_solution,
                                 custom_approx);
  }
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  static_assert(
      tt::assert_conforms_to_v<InterpolationTargetTag,
                               intrp::protocols::InterpolationTargetTag>);
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
  using simple_tags = tmpl::list<Tags::TestTargetPoints>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
  using replace_these_simple_actions =
      tmpl::list<intrp::Actions::InterpolationTargetVarsFromElement<
          InterpolationTargetTag>>;
  using with_these_simple_actions = tmpl::list<
      MockInterpolationTargetVarsFromElement<InterpolationTargetTag>>;
};

template <typename ElemComponent, bool UseTimeDependentMaps,
          typename DomainCreator, typename Runner, typename TemporalId>
std::tuple<Variables<tmpl::list<Tags::TestSolution>>, Mesh<3>>
make_volume_data_and_mesh(const DomainCreator& domain_creator, Runner& runner,
                          const Domain<3>& domain,
                          const ElementId<3>& element_id,
                          const TemporalId& temporal_id) {
  const auto& block = domain.blocks()[element_id.block_id()];
  Mesh<3> mesh{domain_creator.initial_extents()[element_id.block_id()],
               Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};

  const auto inertial_coords =
      [&element_id, &block, &mesh, &runner, &temporal_id]() {
        if constexpr (UseTimeDependentMaps) {
          const auto& functions_of_time = get<domain::Tags::FunctionsOfTime>(
              ActionTesting::cache<ElemComponent>(runner, element_id));
          ElementMap<3, ::Frame::Grid> map_logical_to_grid{
              element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
          return block.moving_mesh_grid_to_inertial_map()(
              map_logical_to_grid(logical_coordinates(mesh)),
              temporal_id.substep_time(), functions_of_time);
        } else {
          (void)runner;
          (void)temporal_id;
          ElementMap<3, Frame::Inertial> map{
              element_id, block.stationary_map().get_clone()};
          return map(logical_coordinates(mesh));
        }
      }();

  // create volume data
  Variables<tmpl::list<Tags::TestSolution>> vars(mesh.number_of_grid_points());
  fill_variables<Tags::TestSolution>(make_not_null(&vars), inertial_coords);
  return std::make_tuple(std::move(vars), mesh);
}

// This is the main test; Takes a functor as an argument so that
// different tests can call it to do slightly different things.
template <typename Metavariables, typename elem_component, typename Functor>
void test_interpolate_on_element(
    Functor initialize_elements_and_queue_simple_actions) {
  using metavars = Metavariables;
  using target_component =
      mock_interpolation_target<metavars,
                                typename metavars::InterpolationTargetA>;

  const auto domain_creator = []() {
    if constexpr (Metavariables::use_time_dependent_maps) {
      return domain::creators::Sphere(
          0.9, 2.9, domain::creators::Sphere::Excision{}, 2_st, 7_st, false,
          std::nullopt, {}, {domain::CoordinateMaps::Distribution::Linear},
          ShellWedges::All,
          std::make_unique<
              domain::creators::time_dependence::UniformTranslation<3>>(
              0.0, std::array<double, 3>({{0.1, 0.2, 0.3}})));
    } else {
      return domain::creators::Sphere(
          0.9, 2.9, domain::creators::Sphere::Excision{}, 2_st, 7_st, false);
    }
  }();
  const auto domain = domain_creator.create_domain();

  Slab slab(0.0, 1.0);
  TimeStepId temporal_id(true, 0, Time(slab, Rational(11, 15)));

  // Create Element_ids.
  const std::vector<ElementId<3>> element_ids =
      initial_element_ids(domain_creator.initial_refinement_levels());

  // This name must match the hard coded one in UniformTranslation
  const std::string f_of_t_name = "Translation";
  std::unordered_map<std::string, double> initial_expiration_times{};
  initial_expiration_times[f_of_t_name] =
      13.5 / 16.0;  // Arbitrary value greater than temporal_id above.

  // Create target points and interp_point_info
  const size_t num_points = 10;
  tnsr::I<DataVector, 3, Frame::Inertial> target_points(num_points);
  const typename intrp::Tags::InterpPointInfo<
      metavars>::type interp_point_info = [&target_points]() {
    MAKE_GENERATOR(gen);
    std::uniform_real_distribution<> r_dist(0.9001, 2.8999);
    std::uniform_real_distribution<> theta_dist(0.0, M_PI);
    std::uniform_real_distribution<> phi_dist(0.0, 2 * M_PI);
    for (size_t i = 0; i < num_points; ++i) {
      const double r = r_dist(gen);
      const double theta = theta_dist(gen);
      const double phi = phi_dist(gen);
      get<0>(target_points)[i] = r * sin(theta) * cos(phi);
      get<1>(target_points)[i] = r * sin(theta) * sin(phi);
      get<2>(target_points)[i] = r * cos(theta);
    }
    typename intrp::Tags::InterpPointInfo<metavars>::type interp_point_info_l{};
    get<intrp::Vars::PointInfoTag<typename metavars::InterpolationTargetA, 3>>(
        interp_point_info_l) = target_points;
    return interp_point_info_l;
  }();

  // Emplace target component.
  auto runner = [&domain_creator, &initial_expiration_times]() {
    if constexpr (Metavariables::use_time_dependent_maps) {
      return ActionTesting::MockRuntimeSystem<metavars>(
          domain_creator.create_domain(),
          domain_creator.functions_of_time(initial_expiration_times));
    } else {
      (void)initial_expiration_times;
      return ActionTesting::MockRuntimeSystem<metavars>(
          domain_creator.create_domain());
    }
  }();

  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_component_and_initialize<target_component>(
      &runner, 0, {target_points});

  static_assert(
      std::is_same_v<typename metavars::InterpolationTargetA::temporal_id::type,
                     double> or
          std::is_same_v<
              typename metavars::InterpolationTargetA::temporal_id::type,
              TimeStepId>,
      "Unsupported temporal_id type");
  if constexpr (std::is_same_v<
                    typename metavars::InterpolationTargetA::temporal_id::type,
                    double>) {
    initialize_elements_and_queue_simple_actions(
        domain_creator, domain, element_ids, interp_point_info, runner,
        temporal_id.substep_time());
  } else if constexpr (std::is_same_v<typename metavars::InterpolationTargetA::
                                          temporal_id::type,
                                      TimeStepId>) {
    initialize_elements_and_queue_simple_actions(domain_creator, domain,
                                                 element_ids, interp_point_info,
                                                 runner, temporal_id);
  }

  // Only some of the actions/events just invoked on elements (those
  // elements which contain target points) will queue a simple action on the
  // InterpolationTarget.  Invoke those simple actions now.
  while (not ActionTesting::is_simple_action_queue_empty<target_component>(
      runner, 0)) {
    runner.template invoke_queued_simple_action<target_component>(0);
  }
}
}  // namespace InterpolateOnElementTestHelpers
