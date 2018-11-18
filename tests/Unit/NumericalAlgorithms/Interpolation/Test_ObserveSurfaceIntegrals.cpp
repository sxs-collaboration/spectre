// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>
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
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/Shell.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Tags.hpp" // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/Callbacks/ObserveSurfaceIntegrals.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp" // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetKerrHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp" // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
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
}  // namespace Tags

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
  struct InterpolationTargetA {
    using compute_items_on_source = tmpl::list<>;
    using vars_to_interpolate_to_target =
        tmpl::list<Tags::TestSolution,
                   gr::Tags::SpatialMetric<3, Frame::Inertial>>;
    using compute_items_on_target = tmpl::list<Tags::SquareComputeItem>;
    using compute_target_points =
        intrp::Actions::KerrHorizon<InterpolationTargetA, ::Frame::Inertial>;
    using post_interpolation_callback =
      intrp::callbacks::ObserveSurfaceIntegrals<
            tmpl::list<Tags::Square>, observers::OptionTags::ReductionFileName,
            Frame::Inertial>;
    // This `type` is so this tag can be used to read options.
    using type = typename compute_target_points::options_type;
  };
  using interpolator_source_vars =
      tmpl::list<Tags::TestSolution,
                 gr::Tags::SpatialMetric<3, Frame::Inertial>>;
  using interpolation_target_tags = tmpl::list<InterpolationTargetA>;
  using temporal_id = TimeId;
  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;
  using component_list = tmpl::list<
      MockInterpolationTarget<MockMetavariables, InterpolationTargetA>,
      MockInterpolator<MockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialize, Exit };
};

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolator.ObserveSurfaceIntegrals", "[Unit]") {
  using metavars = MockMetavariables;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using TupleOfMockDistributedObjects =
      MockRuntimeSystem::TupleOfMockDistributedObjects;
  TupleOfMockDistributedObjects dist_objects{};
  using MockDistributedObjectsTagTargetA =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          MockInterpolationTarget<metavars, metavars::InterpolationTargetA>>;
  using MockDistributedObjectsTagInterpolator =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          MockInterpolator<metavars>>;
  tuples::get<MockDistributedObjectsTagTargetA>(dist_objects)
      .emplace(0,
               ActionTesting::MockDistributedObject<MockInterpolationTarget<
                   metavars, metavars::InterpolationTargetA>>{});
  tuples::get<MockDistributedObjectsTagInterpolator>(dist_objects)
      .emplace(
          0,
          ActionTesting::MockDistributedObject<MockInterpolator<metavars>>{});

  // Options for all InterpolationTargets.
  intrp::OptionHolders::KerrHorizon kerr_horizon_opts(10, {{0.0, 0.0, 0.0}},
                                                      1.0, {{0.0, 0.0, 0.0}});
  std::string h5_file_prefix = "Test_ObserveSurfaceIntegrals";
  tuples::TaggedTuple<metavars::InterpolationTargetA,
                      observers::OptionTags::ReductionFileName>
      tuple_of_opts(kerr_horizon_opts, h5_file_prefix);

  MockRuntimeSystem runner{tuple_of_opts, std::move(dist_objects)};

  const auto domain_creator =
      domain::creators::Shell<Frame::Inertial>(0.9, 4.9, 1, {{5, 5}}, false);

  runner.simple_action<
      MockInterpolationTarget<metavars, metavars::InterpolationTargetA>,
      ::intrp::Actions::InitializeInterpolationTarget<
          metavars::InterpolationTargetA>>(0, domain_creator.create_domain());
  runner.simple_action<MockInterpolator<metavars>,
                       ::intrp::Actions::InitializeInterpolator>(0);

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

  // Tell the InterpolationTargets that we want to interpolate at
  // temporal_id.
  runner.simple_action<
      MockInterpolationTarget<metavars, metavars::InterpolationTargetA>,
      intrp::Actions::AddTemporalIdsToInterpolationTarget<
          metavars::InterpolationTargetA>>(0, std::vector<TimeId>{temporal_id});

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

    // Fill the metric with Minkowski.
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

  // By hand compute integral(r^2 d(cos theta) dphi (2x+3y+5z)^2)
  const double expected_integral = 2432.0 * M_PI / 3.0;
  const std::vector<std::string> expected_legend{"Time", "Square"};

  // Check that the H5 file was written correctly.
  const auto h5_file_name = h5_file_prefix + ".h5";
  const auto file = h5::H5File<h5::AccessType::ReadOnly>(h5_file_name);
  const auto& dat_file = file.get<h5::Dat>("/surface_integrals");
  const Matrix written_data = dat_file.get_data();
  const auto& written_legend = dat_file.get_legend();
  CHECK(written_legend == expected_legend);
  CHECK(0.0 == written_data(0, 0));
  // The interpolation is not perfect because I use too few grid points.
  Approx custom_approx = Approx::custom().epsilon(1.e-4).scale(1.0);
  CHECK(expected_integral == custom_approx(written_data(0, 1)));

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
}  // namespace
