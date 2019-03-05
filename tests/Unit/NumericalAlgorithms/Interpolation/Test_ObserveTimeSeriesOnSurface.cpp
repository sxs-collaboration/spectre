// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/Helpers.hpp"  // IWYU pragma: keep
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"  // IWYU pragma: keep
#include "IO/Observer/Tags.hpp"              // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetKerrHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace StrahlkorperGr {
namespace Tags {
template <typename Frame>
struct AreaElement;
template <typename IntegrandTag, typename Frame>
struct SurfaceIntegral;
}  // namespace Tags
}  // namespace StrahlkorperGr
/// \endcond

namespace {
// Simple DataBoxItems for test.
namespace Tags {
struct TestSolution : db::SimpleTag {
  static std::string name() noexcept { return "TestSolution"; }
  using type = Scalar<DataVector>;
};
struct Square : db::SimpleTag {
  static std::string name() noexcept { return "Square"; }
  using type = Scalar<DataVector>;
};
struct SquareComputeItem : Square, db::ComputeTag {
  static std::string name() noexcept { return "Square"; }
  static Scalar<DataVector> function(const Scalar<DataVector>& x) noexcept {
    auto result = make_with_value<Scalar<DataVector>>(x, 0.0);
    get(result) = square(get(x));
    return result;
  }
  using argument_tags = tmpl::list<TestSolution>;
};
struct Negate : db::SimpleTag {
  static std::string name() noexcept { return "Negate"; }
  using type = Scalar<DataVector>;
};
struct NegateComputeItem : Negate, db::ComputeTag {
  static std::string name() noexcept { return "Negate"; }
  static Scalar<DataVector> function(const Scalar<DataVector>& x) noexcept {
    auto result = make_with_value<Scalar<DataVector>>(x, 0.0);
    get(result) = -get(x);
    return result;
  }
  using argument_tags = tmpl::list<Square>;
};
}  // namespace Tags

template <typename Metavariables>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using action_list = tmpl::list<>;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;
  using initial_databox =
      db::compute_databox_type<typename observers::Actions::InitializeWriter<
          Metavariables>::return_tag_list>;

  using const_global_cache_tag_list =
      tmpl::list<observers::OptionTags::ReductionFileName>;
};

template <typename Metavariables, typename InterpolationTargetTag>
struct MockInterpolationTarget {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using action_list = tmpl::list<>;
  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
  using initial_databox = db::compute_databox_type<
      typename intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template return_tag_list<Metavariables>>;

  using const_global_cache_tag_list = Parallel::get_const_global_cache_tags<
      tmpl::list<typename InterpolationTargetTag::compute_target_points,
                 typename InterpolationTargetTag::post_interpolation_callback>>;
};

template <typename Metavariables>
struct MockInterpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using component_being_mocked = intrp::Interpolator<Metavariables>;
  using initial_databox =
      db::compute_databox_type<typename intrp::Actions::InitializeInterpolator::
                                   template return_tag_list<Metavariables>>;
};

struct MockMetavariables {
  struct SurfaceA {
    using compute_items_on_source = tmpl::list<>;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution,
                   gr::Tags::SpatialMetric<3, Frame::Inertial>>;
    using compute_items_on_target = tmpl::list<
        Tags::SquareComputeItem,
        StrahlkorperGr::Tags::AreaElement<Frame::Inertial>,
        StrahlkorperGr::Tags::SurfaceIntegral<Tags::Square, ::Frame::Inertial>>;
    using compute_target_points =
        intrp::Actions::KerrHorizon<SurfaceA, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<StrahlkorperGr::Tags::SurfaceIntegral<
                Tags::Square, ::Frame::Inertial>>,
            SurfaceA, SurfaceA>;
    // This `type` is so this tag can be used to read options.
    using type = typename compute_target_points::options_type;
  };
  struct SurfaceB {
    using compute_items_on_source = tmpl::list<>;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution,
                   gr::Tags::SpatialMetric<3, Frame::Inertial>>;
    using compute_items_on_target = tmpl::list<
        Tags::SquareComputeItem, Tags::NegateComputeItem,
        StrahlkorperGr::Tags::AreaElement<Frame::Inertial>,
        StrahlkorperGr::Tags::SurfaceIntegral<Tags::Square, Frame::Inertial>,
        StrahlkorperGr::Tags::SurfaceIntegral<Tags::Negate, Frame::Inertial>>;
    using compute_target_points =
        intrp::Actions::KerrHorizon<SurfaceB, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<StrahlkorperGr::Tags::SurfaceIntegral<Tags::Square,
                                                             ::Frame::Inertial>,
                       StrahlkorperGr::Tags::SurfaceIntegral<
                           Tags::Negate, ::Frame::Inertial>>,
            SurfaceB, SurfaceB>;
    // This `type` is so this tag can be used to read options.
    using type = typename compute_target_points::options_type;
  };
  struct SurfaceC {
    using compute_items_on_source = tmpl::list<>;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution,
                   gr::Tags::SpatialMetric<3, Frame::Inertial>>;
    using compute_items_on_target = tmpl::list<
        Tags::SquareComputeItem, Tags::NegateComputeItem,
        StrahlkorperGr::Tags::AreaElement<Frame::Inertial>,
        StrahlkorperGr::Tags::SurfaceIntegral<Tags::Negate, ::Frame::Inertial>>;
    using compute_target_points =
        intrp::Actions::KerrHorizon<SurfaceC, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<
            tmpl::list<StrahlkorperGr::Tags::SurfaceIntegral<
                Tags::Negate, ::Frame::Inertial>>,
            SurfaceC, SurfaceC>;
    // This `type` is so this tag can be used to read options.
    using type = typename compute_target_points::options_type;
  };

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<typename SurfaceA::post_interpolation_callback,
                 typename SurfaceB::post_interpolation_callback,
                 typename SurfaceC::post_interpolation_callback>>;

  using interpolator_source_vars =
      tmpl::list<Tags::TestSolution,
                 gr::Tags::SpatialMetric<3, Frame::Inertial>>;
  using interpolation_target_tags = tmpl::list<SurfaceA, SurfaceB, SurfaceC>;
  using temporal_id = ::Tags::TimeId;
  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;
  using component_list =
      tmpl::list<MockObserverWriter<MockMetavariables>,
                 MockInterpolationTarget<MockMetavariables, SurfaceA>,
                 MockInterpolationTarget<MockMetavariables, SurfaceB>,
                 MockInterpolationTarget<MockMetavariables, SurfaceC>,
                 MockInterpolator<MockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolator.ObserveTimeSeriesOnSurface",
    "[Unit]") {
  const std::string h5_file_prefix = "Test_ObserveTimeSeriesOnSurface";
  const auto h5_file_name = h5_file_prefix + ".h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  using metavars = MockMetavariables;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using TupleOfMockDistributedObjects =
      MockRuntimeSystem::TupleOfMockDistributedObjects;
  TupleOfMockDistributedObjects dist_objects{};
  using MockDistributedObjectsTagTargetA =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          MockInterpolationTarget<metavars, metavars::SurfaceA>>;
  using MockDistributedObjectsTagTargetB =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          MockInterpolationTarget<metavars, metavars::SurfaceB>>;
  using MockDistributedObjectsTagTargetC =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          MockInterpolationTarget<metavars, metavars::SurfaceC>>;
  using MockDistributedObjectsTagInterpolator =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          MockInterpolator<metavars>>;
  using MockDistributedObjectsTagObserverWriter =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          MockObserverWriter<metavars>>;
  tuples::get<MockDistributedObjectsTagTargetA>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<
                      MockInterpolationTarget<metavars, metavars::SurfaceA>>{});
  tuples::get<MockDistributedObjectsTagTargetB>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<
                      MockInterpolationTarget<metavars, metavars::SurfaceB>>{});
  tuples::get<MockDistributedObjectsTagTargetC>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<
                      MockInterpolationTarget<metavars, metavars::SurfaceC>>{});
  tuples::get<MockDistributedObjectsTagInterpolator>(dist_objects)
      .emplace(
          0,
          ActionTesting::MockDistributedObject<MockInterpolator<metavars>>{});
  tuples::get<MockDistributedObjectsTagObserverWriter>(dist_objects)
      .emplace(
          0,
          ActionTesting::MockDistributedObject<MockObserverWriter<metavars>>{});

  // Options for all InterpolationTargets.
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_A(10, {{0.0, 0.0, 0.0}},
                                                        1.0, {{0.0, 0.0, 0.0}});
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_B(10, {{0.0, 0.0, 0.0}},
                                                        2.0, {{0.0, 0.0, 0.0}});
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts_C(10, {{0.0, 0.0, 0.0}},
                                                        1.5, {{0.0, 0.0, 0.0}});
  tuples::TaggedTuple<observers::OptionTags::ReductionFileName,
                      metavars::SurfaceA, metavars::SurfaceB,
                      metavars::SurfaceC>
      tuple_of_opts(h5_file_prefix, std::move(kerr_horizon_opts_A),
                    std::move(kerr_horizon_opts_B),
                    std::move(kerr_horizon_opts_C));

  MockRuntimeSystem runner{std::move(tuple_of_opts), std::move(dist_objects)};

  const auto domain_creator =
      domain::creators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{5, 5}}, false);

  runner.simple_action<
      MockInterpolationTarget<metavars, metavars::SurfaceA>,
      ::intrp::Actions::InitializeInterpolationTarget<metavars::SurfaceA>>(
      0, domain_creator.create_domain());
  runner.simple_action<
      MockInterpolationTarget<metavars, metavars::SurfaceB>,
      ::intrp::Actions::InitializeInterpolationTarget<metavars::SurfaceB>>(
      0, domain_creator.create_domain());
  runner.simple_action<
      MockInterpolationTarget<metavars, metavars::SurfaceC>,
      ::intrp::Actions::InitializeInterpolationTarget<metavars::SurfaceC>>(
      0, domain_creator.create_domain());
  runner.simple_action<MockInterpolator<metavars>,
                       ::intrp::Actions::InitializeInterpolator>(0);
  runner.simple_action<MockObserverWriter<metavars>,
                       ::observers::Actions::InitializeWriter<metavars>>(0);

  Slab slab(0.0, 1.0);
  TimeId temporal_id(true, 0, Time(slab, 0));
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
  // each one.
  for (size_t i = 0; i < element_ids.size(); ++i) {
    runner.simple_action<MockInterpolator<metavars>,
                         intrp::Actions::RegisterElement>(0);
  }

  // Register the InterpolationTargets with the ObserverWriter.
  runner
      .simple_action<MockInterpolationTarget<metavars, metavars::SurfaceA>,
                     ::observers::Actions::RegisterSingletonWithObserverWriter>(
          0, observers::ObservationId(temporal_id.time().value(),
                                      typename metavars::SurfaceA{}));
  runner
      .simple_action<MockInterpolationTarget<metavars, metavars::SurfaceB>,
                     ::observers::Actions::RegisterSingletonWithObserverWriter>(
          0, observers::ObservationId(temporal_id.time().value(),
                                      typename metavars::SurfaceB{}));
  runner
      .simple_action<MockInterpolationTarget<metavars, metavars::SurfaceC>,
                     ::observers::Actions::RegisterSingletonWithObserverWriter>(
          0, observers::ObservationId(temporal_id.time().value(),
                                      typename metavars::SurfaceC{}));

  // Tell the InterpolationTargets that we want to interpolate at
  // temporal_id.
  runner.simple_action<
      MockInterpolationTarget<metavars, metavars::SurfaceA>,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<metavars::SurfaceA>>(
      0, std::vector<TimeId>{temporal_id});
  runner.simple_action<
      MockInterpolationTarget<metavars, metavars::SurfaceB>,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<metavars::SurfaceB>>(
      0, std::vector<TimeId>{temporal_id});
  runner.simple_action<
      MockInterpolationTarget<metavars, metavars::SurfaceC>,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<metavars::SurfaceC>>(
      0, std::vector<TimeId>{temporal_id});

  // There should be three queued simple actions (registration), so invoke
  // them and check that there are no more.
  runner.invoke_queued_simple_action<MockObserverWriter<metavars>>(0);
  runner.invoke_queued_simple_action<MockObserverWriter<metavars>>(0);
  runner.invoke_queued_simple_action<MockObserverWriter<metavars>>(0);
  CHECK(runner.is_simple_action_queue_empty<MockObserverWriter<metavars>>(0));

  // Create volume data and send it to the interpolator.
  for (const auto& element_id : element_ids) {
    const auto& block = domain.blocks()[element_id.block_id()];
    ::Mesh<3> mesh{domain_creator.initial_extents()[element_id.block_id()],
                   Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    ElementMap<3, Frame::Inertial> map{element_id,
                                       block.coordinate_map().get_clone()};
    const auto inertial_coords = map(logical_coordinates(mesh));
    db::item_type<
        ::Tags::Variables<typename metavars::interpolator_source_vars>>
        output_vars(mesh.number_of_grid_points());
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
    runner.simple_action<MockInterpolator<metavars>,
                         intrp::Actions::InterpolatorReceiveVolumeData>(
        0, temporal_id, element_id, mesh, std::move(output_vars));
  }

  // Invoke remaining actions in random order.
  MAKE_GENERATOR(generator);
  auto index_map = ActionTesting::indices_of_components_with_queued_actions<
      metavars::component_list>(make_not_null(&runner), 0_st);
  while (not index_map.empty()) {
    ActionTesting::invoke_random_queued_action<metavars::component_list>(
        make_not_null(&runner), make_not_null(&generator), index_map, 0_st);
    index_map = ActionTesting::indices_of_components_with_queued_actions<
        metavars::component_list>(make_not_null(&runner), 0_st);
  }

  // There should be three more threaded actions, so invoke them and check
  // that there are no more.
  runner.invoke_queued_threaded_action<MockObserverWriter<metavars>>(0);
  runner.invoke_queued_threaded_action<MockObserverWriter<metavars>>(0);
  runner.invoke_queued_threaded_action<MockObserverWriter<metavars>>(0);
  CHECK(runner.is_threaded_action_queue_empty<MockObserverWriter<metavars>>(0));

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
                                                   "SurfaceIntegralSquare"};
  const std::vector<std::string> expected_legend_b{
      "Time", "SurfaceIntegralSquare", "SurfaceIntegralNegate"};
  const std::vector<std::string> expected_legend_c{"Time",
                                                   "SurfaceIntegralNegate"};

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
