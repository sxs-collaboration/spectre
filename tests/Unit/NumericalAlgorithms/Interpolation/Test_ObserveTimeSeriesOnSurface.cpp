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

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Actions/RegisterSingleton.hpp"
#include "IO/Observer/Helpers.hpp"  // IWYU pragma: keep
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"              // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolationTargetKerrHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
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
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace StrahlkorperGr::Tags {
template <typename Frame>
struct AreaElement;
template <typename IntegrandTag, typename Frame>
struct SurfaceIntegral;
}  // namespace StrahlkorperGr::Tags

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
                       const Scalar<DataVector>& x) noexcept {
    get(*result) = square(get(x));
  }
  using argument_tags = tmpl::list<TestSolution>;
  using base = Square;
  using return_type = Scalar<DataVector>;
};
struct Negate : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct NegateCompute : Negate, db::ComputeTag {
  static void function(gsl::not_null<Scalar<DataVector>*> result,
                       const Scalar<DataVector>& x) noexcept {
    get(*result) = -get(x);
  }
  using argument_tags = tmpl::list<Square>;
  using base = Negate;
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
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<Actions::SetupDataBox,
                     observers::Actions::InitializeWriter<Metavariables>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Registration, tmpl::list<>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;

  using component_being_mocked = observers::ObserverWriter<Metavariables>;
};

template <typename Metavariables, typename InterpolationTargetTag>
struct MockInterpolationTarget {
 private:
  struct RegistrationHelper {
    template <typename ParallelComponent, typename DbTagsList,
              typename ArrayIndex>
    static std::pair<observers::TypeOfObservation, observers::ObservationKey>
    register_info(const db::DataBox<DbTagsList>& /*box*/,
                  const ArrayIndex& /*array_index*/) noexcept {
      return {observers::TypeOfObservation::Reduction,
              observers::ObservationKey{
                  pretty_type::get_name<InterpolationTargetTag>()}};
    }
  };

 public:
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
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<Actions::SetupDataBox,
                     intrp::Actions::InitializeInterpolationTarget<
                         Metavariables, InterpolationTargetTag>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Registration,
          tmpl::list<::observers::Actions::RegisterSingletonWithObserverWriter<
              RegistrationHelper>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
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
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<Actions::SetupDataBox,
                     ::intrp::Actions::InitializeInterpolator<
                         intrp::Tags::VolumeVarsInfo<Metavariables,
                                                     ::Tags::TimeStepId>,
                         intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Registration, tmpl::list<>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing, tmpl::list<>>>;
  using component_being_mocked = intrp::Interpolator<Metavariables>;
};

struct MockMetavariables {
  struct SurfaceA {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution,
                   gr::Tags::SpatialMetric<3, Frame::Inertial>>;
    using compute_items_on_target =
        tmpl::list<Tags::SquareCompute,
                   StrahlkorperGr::Tags::AreaElementCompute<Frame::Inertial>,
                   StrahlkorperGr::Tags::SurfaceIntegralCompute<
                       Tags::Square, ::Frame::Inertial>>;
    using compute_target_points =
        intrp::TargetPoints::KerrHorizon<SurfaceA, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<StrahlkorperGr::Tags::SurfaceIntegral<
                Tags::Square, ::Frame::Inertial>>,
            SurfaceA, SurfaceA>;
  };
  struct SurfaceB {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution,
                   gr::Tags::SpatialMetric<3, Frame::Inertial>>;
    using compute_items_on_target =
        tmpl::list<Tags::SquareCompute, Tags::NegateCompute,
                   StrahlkorperGr::Tags::AreaElementCompute<Frame::Inertial>,
                   StrahlkorperGr::Tags::SurfaceIntegralCompute<
                       Tags::Square, Frame::Inertial>,
                   StrahlkorperGr::Tags::SurfaceIntegralCompute<
                       Tags::Negate, Frame::Inertial>>;
    using compute_target_points =
        intrp::TargetPoints::KerrHorizon<SurfaceB, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<StrahlkorperGr::Tags::SurfaceIntegral<Tags::Square,
                                                             ::Frame::Inertial>,
                       StrahlkorperGr::Tags::SurfaceIntegral<
                           Tags::Negate, ::Frame::Inertial>>,
            SurfaceB, SurfaceB>;
  };
  struct SurfaceC {
    using temporal_id = ::Tags::TimeStepId;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution,
                   gr::Tags::SpatialMetric<3, Frame::Inertial>>;
    using compute_items_on_target =
        tmpl::list<Tags::SquareCompute, Tags::NegateCompute,
                   StrahlkorperGr::Tags::AreaElementCompute<Frame::Inertial>,
                   StrahlkorperGr::Tags::SurfaceIntegralCompute<
                       Tags::Negate, ::Frame::Inertial>>;
    using compute_target_points =
        intrp::TargetPoints::KerrHorizon<SurfaceC, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<StrahlkorperGr::Tags::SurfaceIntegral<
                Tags::Negate, ::Frame::Inertial>>,
            SurfaceC, SurfaceC>;
  };

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<typename SurfaceA::post_interpolation_callback,
                 typename SurfaceB::post_interpolation_callback,
                 typename SurfaceC::post_interpolation_callback>>;

  using interpolator_source_vars =
      tmpl::list<Tags::TestSolution,
                 gr::Tags::SpatialMetric<3, Frame::Inertial>>;
  using interpolation_target_tags = tmpl::list<SurfaceA, SurfaceB, SurfaceC>;
  static constexpr size_t volume_dim = 3;
  using component_list =
      tmpl::list<MockObserverWriter<MockMetavariables>,
                 MockInterpolationTarget<MockMetavariables, SurfaceA>,
                 MockInterpolationTarget<MockMetavariables, SurfaceB>,
                 MockInterpolationTarget<MockMetavariables, SurfaceC>,
                 MockInterpolator<MockMetavariables>>;
  enum class Phase { Initialization, Registration, Testing, Exit };
};

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolator.ObserveTimeSeriesOnSurface",
    "[Unit]") {
  domain::creators::register_derived_with_charm();

  const std::string h5_file_prefix = "Test_ObserveTimeSeriesOnSurface";
  const auto h5_file_name = h5_file_prefix + ".h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  using metavars = MockMetavariables;
  using interp_component = MockInterpolator<metavars>;
  using target_a_component =
      MockInterpolationTarget<metavars, metavars::SurfaceA>;
  using target_b_component =
      MockInterpolationTarget<metavars, metavars::SurfaceB>;
  using target_c_component =
      MockInterpolationTarget<metavars, metavars::SurfaceC>;
  using obs_writer = MockObserverWriter<metavars>;

  // Options for all InterpolationTargets.
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_A(10, {{0.0, 0.0, 0.0}},
                                                        1.0, {{0.0, 0.0, 0.0}});
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_B(10, {{0.0, 0.0, 0.0}},
                                                        2.0, {{0.0, 0.0, 0.0}});
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_C(10, {{0.0, 0.0, 0.0}},
                                                        1.5, {{0.0, 0.0, 0.0}});
  const auto domain_creator =
      domain::creators::Shell(0.9, 4.9, 1, {{5, 5}}, false);
  tuples::TaggedTuple<observers::Tags::ReductionFileName,
                      ::intrp::Tags::KerrHorizon<metavars::SurfaceA>,
                      domain::Tags::Domain<3>,
                      ::intrp::Tags::KerrHorizon<metavars::SurfaceB>,
                      ::intrp::Tags::KerrHorizon<metavars::SurfaceC>>
      tuple_of_opts{h5_file_prefix, kerr_horizon_opts_A,
                    domain_creator.create_domain(), kerr_horizon_opts_B,
                    kerr_horizon_opts_C};

  // Three mock nodes, with 2, 1, and 3 mock cores.
  ActionTesting::MockRuntimeSystem<metavars> runner{
      std::move(tuple_of_opts), {}, {2, 1, 3}};

  ActionTesting::set_phase(make_not_null(&runner),
                           metavars::Phase::Initialization);
  ActionTesting::emplace_group_component<interp_component>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t core = 0; core < 6; ++core) {
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
      &runner, ActionTesting::NodeId{2}, ActionTesting::LocalCoreId{2});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_b_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_singleton_component<target_c_component>(
      &runner, ActionTesting::NodeId{2}, ActionTesting::LocalCoreId{1});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_c_component>(make_not_null(&runner), 0);
  }
  ActionTesting::emplace_nodegroup_component<obs_writer>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t node = 0; node < 2; ++node) {
      ActionTesting::next_action<obs_writer>(make_not_null(&runner), node);
    }
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           metavars::Phase::Registration);

  Slab slab(0.0, 1.0);
  TimeStepId temporal_id(true, 0, Time(slab, 0));
  const auto domain = domain_creator.create_domain();

  // Create element_ids.
  std::vector<ElementId<3>> element_ids{};
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
  std::unordered_map<ElementId<3>, size_t> mock_core_for_each_element;
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

  // Register the InterpolationTargets with the ObserverWriter.
  ActionTesting::next_action<target_a_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<target_b_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<target_c_component>(make_not_null(&runner), 0);

  // Tell the InterpolationTargets that we want to interpolate at
  // temporal_id.
  ActionTesting::simple_action<
      target_a_component,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<metavars::SurfaceA>>(
      make_not_null(&runner), 0, std::vector<TimeStepId>{temporal_id});
  ActionTesting::simple_action<
      target_b_component,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<metavars::SurfaceB>>(
      make_not_null(&runner), 0, std::vector<TimeStepId>{temporal_id});
  ActionTesting::simple_action<
      target_c_component,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<metavars::SurfaceC>>(
      make_not_null(&runner), 0, std::vector<TimeStepId>{temporal_id});

  // There should be three queued simple actions (registration), so invoke
  // them and check that there are no more.
  // RegisterSingletonWithObserverWriter should have placed these simple queued
  // actions on node 0, so that's where we must invoke them.
  ActionTesting::invoke_queued_simple_action<obs_writer>(make_not_null(&runner),
                                                         0);
  ActionTesting::invoke_queued_simple_action<obs_writer>(make_not_null(&runner),
                                                         0);
  ActionTesting::invoke_queued_simple_action<obs_writer>(make_not_null(&runner),
                                                         0);
  CHECK(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner, 0));
  CHECK(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner, 1));
  CHECK(ActionTesting::is_simple_action_queue_empty<obs_writer>(runner, 2));

  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  // Create volume data and send it to the interpolator.
  for (const auto& element_id : element_ids) {
    const auto& block = domain.blocks()[element_id.block_id()];
    ::Mesh<3> mesh{domain_creator.initial_extents()[element_id.block_id()],
                   Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    if (block.is_time_dependent()) {
      ERROR("The block must be time-independent");
    }
    ElementMap<3, Frame::Inertial> map{element_id,
                                       block.stationary_map().get_clone()};
    const auto inertial_coords = map(logical_coordinates(mesh));
    ::Variables<typename metavars::interpolator_source_vars> output_vars(
        mesh.number_of_grid_points());
    auto& test_solution = get<Tags::TestSolution>(output_vars);

    // Fill test_solution with some analytic solution.
    get(test_solution) = 2.0 * get<0>(inertial_coords) +
                         3.0 * get<1>(inertial_coords) +
                         5.0 * get<2>(inertial_coords);

    // Fill the metric with Minkowski for simplicity.  The
    // InterpolationTarget is called "KerrHorizon" merely because the
    // surface corresponds to where the horizon *would be* in a Kerr
    // spacetime in Kerr-Schild coordinates; this in no way requires
    // that there is an actual horizon or that the metric is Kerr.
    gr::Solutions::Minkowski<3> solution;
    get<gr::Tags::SpatialMetric<3, Frame::Inertial>>(output_vars) =
        get<gr::Tags::SpatialMetric<3, Frame::Inertial>>(solution.variables(
            inertial_coords, 0.0,
            tmpl::list<gr::Tags::SpatialMetric<3, Frame::Inertial>>{}));

    // Call the InterpolatorReceiveVolumeData action on each element_id.
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::InterpolatorReceiveVolumeData<
                                     typename metavars::SurfaceA::temporal_id>>(
        make_not_null(&runner), mock_core_for_each_element.at(element_id),
        temporal_id, element_id, mesh, std::move(output_vars));
  }

  // Invoke remaining actions in random order.
  MAKE_GENERATOR(generator);
  auto array_indices_with_queued_simple_actions =
      ActionTesting::array_indices_with_queued_simple_actions<
          metavars::component_list>(make_not_null(&runner));
  while (ActionTesting::number_of_elements_with_queued_simple_actions<
             metavars::component_list>(
             array_indices_with_queued_simple_actions) > 0) {
    ActionTesting::invoke_random_queued_simple_action<metavars::component_list>(
        make_not_null(&runner), make_not_null(&generator),
        array_indices_with_queued_simple_actions);
    array_indices_with_queued_simple_actions =
        ActionTesting::array_indices_with_queued_simple_actions<
            metavars::component_list>(make_not_null(&runner));
  }

  // There should be three more threaded actions, so invoke them and check
  // that there are no more.  They should all be on node zero.
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);
  CHECK(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 0));
  CHECK(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 1));
  CHECK(ActionTesting::is_threaded_action_queue_empty<obs_writer>(runner, 2));

  // By hand compute integral(r^2 d(cos theta) dphi (2x+3y+5z)^2)
  const std::vector<double> expected_integral_a{2432.0 * M_PI / 3.0};
  // SurfaceB has a larger radius by a factor of 2 than SurfaceA,
  // but the same function.  This results in a factor of 4 increase
  // (because the integrand scales like r^2), and an additional factor of
  // 4 (from the area element), for a net factor of 16.  There is also a
  // minus sign for "Negate".
  const std::vector<double> expected_integral_b{16.0 * 2432.0 * M_PI / 3.0,
                                                -16.0 * 2432.0 * M_PI / 3.0};
  // SurfaceC has a larger radius by a factor of 1.5 than SurfaceA,
  // but the same function.  This results in a factor of 2.25 increase
  // (because the integrand scales like r^2), and an additional factor of
  // 2.25 (from the area element), for a net factor of 5.0625.  There is also a
  // minus sign for "Negate".
  const std::vector<double> expected_integral_c{-5.0625 * 2432.0 * M_PI / 3.0};
  const std::vector<std::string> expected_legend_a{"Time",
                                                   "SurfaceIntegral(Square)"};
  const std::vector<std::string> expected_legend_b{
      "Time", "SurfaceIntegral(Square)", "SurfaceIntegral(Negate)"};
  const std::vector<std::string> expected_legend_c{"Time",
                                                   "SurfaceIntegral(Negate)"};

  // Check that the H5 file was written correctly.
  const auto file = h5::H5File<h5::AccessType::ReadOnly>(h5_file_name);
  auto check_file_contents = [&file](
      const std::vector<double>& expected_integral,
      const std::vector<std::string>& expected_legend,
      const std::string& group_name) noexcept {
    const auto& dat_file = file.get<h5::Dat>(group_name);
    const Matrix written_data = dat_file.get_data();
    const auto& written_legend = dat_file.get_legend();
    CHECK(written_legend == expected_legend);
    CHECK(0.0 == written_data(0, 0));
    // The interpolation is not perfect because I use too few grid points.
    Approx custom_approx = Approx::custom().epsilon(1.e-4).scale(1.0);
    for (size_t i = 0; i < expected_integral.size(); ++i) {
      CHECK(expected_integral[i] == custom_approx(written_data(0, i + 1)));
    }
  };
  check_file_contents(expected_integral_a, expected_legend_a, "/SurfaceA");
  check_file_contents(expected_integral_b, expected_legend_b, "/SurfaceB");
  check_file_contents(expected_integral_c, expected_legend_c, "/SurfaceC");
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
}  // namespace
