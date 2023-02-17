// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/Disk.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/IO/VolumeData.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/AddTemporalIdsToInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/CleanUpInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetReceiveVars.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceivePoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceiveVolumeData.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/TryToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveLineSegment.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/LineSegment.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

// Simple DataBoxItems for test.
namespace Tags {
struct TestSolution : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Square : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct SquareCompute : Square, db::ComputeTag {
  static void function(gsl::not_null<Scalar<DataVector>*> result,
                       const Scalar<DataVector>& x) {
    get(*result) = square(get(x));
  }
  using argument_tags = tmpl::list<TestSolution>;
  using base = Square;
  using return_type = Scalar<DataVector>;
};
}  // namespace Tags

template <typename Metavariables>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = size_t;
  using const_global_cache_tags =
      tmpl::list<observers::Tags::ReductionFileName>;
  using simple_tags =
      typename observers::Actions::InitializeWriter<Metavariables>::simple_tags;
  using compute_tags = typename observers::Actions::InitializeWriter<
      Metavariables>::compute_tags;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<observers::Actions::InitializeWriter<Metavariables>>>,
      Parallel::PhaseActions<Parallel::Phase::Register, tmpl::list<>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
};

template <typename Metavariables, typename InterpolationTargetTag>
struct MockInterpolationTarget {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = size_t;
  using const_global_cache_tags = tmpl::flatten<tmpl::append<
      Parallel::get_const_global_cache_tags_from_actions<tmpl::list<
          typename InterpolationTargetTag::compute_target_points,
          typename InterpolationTargetTag::post_interpolation_callback>>,
      tmpl::list<domain::Tags::Domain<Metavariables::volume_dim>>>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolationTarget<
              Metavariables, InterpolationTargetTag>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
};

template <typename Metavariables>
struct MockInterpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<::intrp::Actions::InitializeInterpolator<
              tmpl::list<
                  intrp::Tags::VolumeVarsInfo<Metavariables,
                                              ::Tags::TimeStepId>,
                  intrp::Tags::VolumeVarsInfo<Metavariables, ::Tags::Time>>,
              intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>,
      Parallel::PhaseActions<Parallel::Phase::Register, tmpl::list<>>,
      Parallel::PhaseActions<Parallel::Phase::Testing, tmpl::list<>>>;
  using component_being_mocked = intrp::Interpolator<Metavariables>;
};

template <size_t Dim>
struct MockMetavariables {
  static constexpr size_t volume_dim = Dim;

  struct LineA : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::Time;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution,
                   gr::Tags::SpatialMetric<volume_dim, Frame::Inertial>,
                   domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
    using compute_items_on_target = tmpl::list<Tags::SquareCompute>;
    using compute_target_points =
        intrp::TargetPoints::LineSegment<LineA, volume_dim>;
    using post_interpolation_callback = intrp::callbacks::ObserveLineSegment<
        tmpl::append<vars_to_interpolate_to_target, compute_items_on_target>,
        LineA>;
  };

  struct LineB : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution,
                   gr::Tags::SpatialMetric<volume_dim, Frame::Inertial>,
                   domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
    using compute_items_on_target = tmpl::list<Tags::SquareCompute>;
    using compute_target_points =
        intrp::TargetPoints::LineSegment<LineB, volume_dim>;
    using post_interpolation_callback = intrp::callbacks::ObserveLineSegment<
        tmpl::append<vars_to_interpolate_to_target, compute_items_on_target>,
        LineB>;
  };

  using observed_reduction_data_tags = tmpl::list<>;

  using interpolator_source_vars =
      tmpl::list<Tags::TestSolution,
                 gr::Tags::SpatialMetric<volume_dim, Frame::Inertial>,
                 domain::Tags::Coordinates<volume_dim, Frame::Inertial>>;
  using interpolation_target_tags = tmpl::list<LineA, LineB>;
  using component_list =
      tmpl::list<MockObserverWriter<MockMetavariables>,
                 MockInterpolationTarget<MockMetavariables, LineA>,
                 MockInterpolationTarget<MockMetavariables, LineB>,
                 MockInterpolator<MockMetavariables>>;
};

// test function which will be interpolated
template <size_t Dim>
DataVector test_function(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords) {
  DataVector res = sin(coords.get(0));
  if constexpr (Dim > 1) {
    res += cos(coords.get(1));
  }
  if constexpr (Dim > 2) {
    res += 3.5 * coords.get(2);
  }
  return res;
}

template <size_t Dim, typename Generator, typename Spacetime>
void run_test(gsl::not_null<Generator*> generator,
              const intrp::OptionHolders::LineSegment<Dim>& line_segment_opts_A,
              const intrp::OptionHolders::LineSegment<Dim>& line_segment_opts_B,
              const DomainCreator<Dim>& domain_creator,
              const Spacetime& spacetime) {
  // Check if either file generated by this test exists and remove them
  // if so. Check for both files existing before the test runs, since
  // both files get written when evaluating the list of post interpolation
  // callbacks below.
  const std::string h5_file_prefix = "Test_ObserveLineSegment";
  const auto h5_file_name = h5_file_prefix + ".h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  using metavars = MockMetavariables<Dim>;

  // Test That ObserveTimeSeriesOnSurface indeed does conform to its protocol
  using callback_A = typename metavars::LineA::post_interpolation_callback;
  using callback_B = typename metavars::LineB::post_interpolation_callback;
  using protocol = intrp::protocols::PostInterpolationCallback;
  static_assert(tt::assert_conforms_to_v<callback_A, protocol>);
  static_assert(tt::assert_conforms_to_v<callback_B, protocol>);

  using interp_component = MockInterpolator<metavars>;
  using target_a_component =
      MockInterpolationTarget<metavars, typename metavars::LineA>;
  using target_b_component =
      MockInterpolationTarget<metavars, typename metavars::LineB>;
  using obs_writer = MockObserverWriter<metavars>;

  tuples::TaggedTuple<observers::Tags::ReductionFileName,
                      ::intrp::Tags::LineSegment<typename metavars::LineA, Dim>,
                      ::intrp::Tags::LineSegment<typename metavars::LineB, Dim>,
                      domain::Tags::Domain<Dim>>
      tuple_of_opts{h5_file_prefix, line_segment_opts_A, line_segment_opts_B,
                    domain_creator.create_domain()};

  // Three mock nodes, with 2, 1, and 4 mock cores.
  ActionTesting::MockRuntimeSystem<metavars> runner{
      std::move(tuple_of_opts), {}, {2, 1, 4}};

  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_group_component<interp_component>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t core = 0; core < 7; ++core) {
      ActionTesting::next_action<interp_component>(make_not_null(&runner),
                                                   core);
    }
  }

  ActionTesting::emplace_singleton_component<target_a_component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{1});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_a_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_singleton_component<target_b_component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{1});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_b_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_nodegroup_component<obs_writer>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t node = 0; node < 2; ++node) {
      ActionTesting::next_action<obs_writer>(make_not_null(&runner), node);
    }
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Register);

  Slab slab(0.0, 1.0);
  TimeStepId temporal_id(true, 0, Time(slab, 0));
  const auto domain = domain_creator.create_domain();

  // Create element_ids.
  std::vector<ElementId<Dim>> element_ids{};
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator.initial_refinement_levels()[block.id()];
    auto elem_ids = initial_element_ids(block.id(), initial_ref_levs);
    element_ids.insert(element_ids.end(), elem_ids.begin(), elem_ids.end());
  }

  // Tell the interpolator how many elements there are by registering
  // each one. Normally intrp::Actions::RegisterElement is called by
  // RegisterElementWithInterpolator, and invoked on the ckLocalBranch
  // of the interpolator that is associated with each element
  // (i.e. the local core on each element).
  // Here we assign elements round-robin to the mock cores.
  // And for group components, the array_index is the global core index.
  const size_t num_cores = runner.num_global_cores();
  std::unordered_map<ElementId<Dim>, size_t> mock_core_for_each_element;
  size_t core_for_next_element = 0;
  for (const auto& element_id : element_ids) {
    mock_core_for_each_element.insert({element_id, core_for_next_element});
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::RegisterElement>(
        make_not_null(&runner), core_for_next_element);
    if (++core_for_next_element >= num_cores) {
      core_for_next_element = 0;
    }
  }

  // Tell the InterpolationTargets that we want to interpolate at
  // temporal_id.
  ActionTesting::simple_action<
      target_a_component, intrp::Actions::AddTemporalIdsToInterpolationTarget<
                              typename metavars::LineA>>(
      make_not_null(&runner), 0,
      std::vector<double>{temporal_id.substep_time().value()});
  ActionTesting::simple_action<
      target_b_component, intrp::Actions::AddTemporalIdsToInterpolationTarget<
                              typename metavars::LineB>>(
      make_not_null(&runner), 0, std::vector<TimeStepId>{temporal_id});

  CHECK(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner, 0));
  CHECK(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner, 1));
  CHECK(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner, 2));

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Create volume data and send it to the interpolator.
  for (const auto& element_id : element_ids) {
    const auto& block = domain.blocks()[element_id.block_id()];
    ::Mesh<Dim> mesh{domain_creator.initial_extents()[element_id.block_id()],
                     Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
    if (block.is_time_dependent()) {
      ERROR("The block must be time-independent");
    }
    ElementMap<Dim, Frame::Inertial> map{element_id,
                                         block.stationary_map().get_clone()};
    const auto inertial_coords = map(logical_coordinates(mesh));
    ::Variables<typename metavars::interpolator_source_vars> output_vars(
        mesh.number_of_grid_points());
    get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(output_vars) =
        inertial_coords;
    get<>(get<Tags::TestSolution>(output_vars)) =
        test_function(inertial_coords);

    get<gr::Tags::SpatialMetric<Dim, Frame::Inertial>>(output_vars) =
        get<gr::Tags::SpatialMetric<Dim, Frame::Inertial>>(spacetime.variables(
            inertial_coords, 0.0,
            tmpl::list<gr::Tags::SpatialMetric<Dim, Frame::Inertial>>{}));

    // Call the InterpolatorReceiveVolumeData action on each element_id.
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::InterpolatorReceiveVolumeData<
                                     typename metavars::LineA::temporal_id>>(
        make_not_null(&runner), mock_core_for_each_element.at(element_id),
        temporal_id.substep_time().value(), element_id, mesh, output_vars);
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::InterpolatorReceiveVolumeData<
                                     typename metavars::LineB::temporal_id>>(
        make_not_null(&runner), mock_core_for_each_element.at(element_id),
        temporal_id, element_id, mesh, std::move(output_vars));
  }

  // Invoke remaining actions in random order.
  auto array_indices_with_queued_simple_actions =
      ActionTesting::array_indices_with_queued_simple_actions<
          typename metavars::component_list>(make_not_null(&runner));

  while (ActionTesting::number_of_elements_with_queued_simple_actions<
             typename metavars::component_list>(
             array_indices_with_queued_simple_actions) > 0) {
    ActionTesting::invoke_random_queued_simple_action<
        typename metavars::component_list>(
        make_not_null(&runner), generator,
        array_indices_with_queued_simple_actions);
    array_indices_with_queued_simple_actions =
        ActionTesting::array_indices_with_queued_simple_actions<
            typename metavars::component_list>(make_not_null(&runner));
  }
  // There should be 2 more threaded actions, so invoke them and check
  // that there are no more.  They should all be on node zero.
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);

  CHECK(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 0));
  CHECK(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 1));
  CHECK(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 2));

  const auto file = h5::H5File<h5::AccessType::ReadOnly>(h5_file_name);

  auto check_file_contents = [&file, &spacetime](const std::string& group_name,
                                                 const tnsr::I<DataVector, Dim>&
                                                     interpolated_coords) {
    file.close_current_object();
    const auto& vol_file = file.get<h5::VolumeData>(group_name);
    const auto& obs_ids = vol_file.list_observation_ids();
    CHECK(obs_ids.size() == 1);
    const auto& obs_value = vol_file.get_observation_value(obs_ids.at(0));
    CHECK(obs_value == 0.);

    // error due to low resolution of domain
    Approx custom_approx = Approx::custom().epsilon(1.e-4).scale(1.0);

    for (size_t i = 0; i < interpolated_coords.size(); ++i) {
      const auto& written_component = vol_file.get_tensor_component(
          obs_ids.at(0),
          "InertialCoordinates" + interpolated_coords.component_suffix(i));
      const auto& written_dv = std::get<DataVector>(written_component.data);
      CHECK_ITERABLE_CUSTOM_APPROX(written_dv, interpolated_coords.get(i),
                                   custom_approx);
    }

    const auto interpolated_metric =
        get<gr::Tags::SpatialMetric<Dim, Frame::Inertial>>(spacetime.variables(
            interpolated_coords, 0.0,
            tmpl::list<gr::Tags::SpatialMetric<Dim, Frame::Inertial>>{}));
    for (size_t i = 0; i < interpolated_metric.size(); ++i) {
      const auto& written_component = vol_file.get_tensor_component(
          obs_ids.at(0),
          "SpatialMetric" + interpolated_metric.component_suffix(i));
      const auto& written_dv = std::get<DataVector>(written_component.data);
      CHECK_ITERABLE_CUSTOM_APPROX(written_dv, interpolated_metric[i],
                                   custom_approx);
    }

    const auto interpolated_test_solution = test_function(interpolated_coords);

    const auto& written_test_solution_component =
        vol_file.get_tensor_component(obs_ids.at(0), "TestSolution");
    const auto& written_test_solution_dv =
        std::get<DataVector>(written_test_solution_component.data);

    CHECK_ITERABLE_CUSTOM_APPROX(written_test_solution_dv,
                                 interpolated_test_solution, custom_approx);

    const auto interpolated_square = square(interpolated_test_solution);
    const auto& written_square_component =
        vol_file.get_tensor_component(obs_ids.at(0), "Square");
    const auto& written_square_dv =
        std::get<DataVector>(written_square_component.data);

    CHECK_ITERABLE_CUSTOM_APPROX(written_square_dv, interpolated_square,
                                 custom_approx);
  };

  const auto& data_box_a =
      ActionTesting::get_databox<target_a_component>(runner, 0);
  const auto interpolated_coords_a =
      intrp::TargetPoints::LineSegment<typename metavars::LineA, Dim>::points(
          data_box_a, tmpl::type_<metavars>{});
  const auto& data_box_b =
      ActionTesting::get_databox<target_b_component>(runner, 0);
  const auto interpolated_coords_b =
      intrp::TargetPoints::LineSegment<typename metavars::LineB, Dim>::points(
          data_box_b, tmpl::type_<metavars>{});

  check_file_contents("/LineA", interpolated_coords_a);
  check_file_contents("/LineB", interpolated_coords_b);

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.ObserveLineSegment",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  MAKE_GENERATOR(generator);

  const auto interval =
      domain::creators::Interval({{0.}}, {{4.}}, {{1}}, {{12}}, {{true}});
  const auto disk = domain::creators::Disk(0.9, 4.9, 1, {{12, 12}}, false);
  const auto shell = domain::creators::Shell(0.9, 4.9, 1, {{12, 12}}, false);

  intrp::OptionHolders::LineSegment<1> line_segment_opts_A_1d({{0.0}}, {{1.0}},
                                                              10);
  intrp::OptionHolders::LineSegment<1> line_segment_opts_B_1d({{2.2}}, {{3.1}},
                                                              10);
  intrp::OptionHolders::LineSegment<2> line_segment_opts_A_2d({{0.0, 1.0}},
                                                              {{0.0, 2.0}}, 10);
  intrp::OptionHolders::LineSegment<2> line_segment_opts_B_2d({{1.0, 2.0}},
                                                              {{2.0, 3.1}}, 10);
  intrp::OptionHolders::LineSegment<3> line_segment_opts_A_3d(
      {{0.0, 0.0, 1.0}}, {{0.0, 0.0, 2.0}}, 10);
  intrp::OptionHolders::LineSegment<3> line_segment_opts_B_3d(
      {{1.3, 1.0, 2.0}}, {{1.7, 2.0, 3.1}}, 10);

  gr::Solutions::Minkowski<1> minkowski_1d{};
  gr::Solutions::Minkowski<2> minkowski_2d{};
  gr::Solutions::Minkowski<3> minkowski_3d{};
  gr::Solutions::KerrSchild kerr_schild{1., {0.3, 0.4, 0.1}, {0., 0., 0.}};

  run_test(make_not_null(&generator), line_segment_opts_A_1d,
           line_segment_opts_B_1d, interval, minkowski_1d);
  run_test(make_not_null(&generator), line_segment_opts_A_2d,
           line_segment_opts_B_2d, disk, minkowski_2d);
  run_test(make_not_null(&generator), line_segment_opts_A_3d,
           line_segment_opts_B_3d, shell, minkowski_3d);
  run_test(make_not_null(&generator), line_segment_opts_A_3d,
           line_segment_opts_B_3d, shell, kerr_schild);
}
}  // namespace
