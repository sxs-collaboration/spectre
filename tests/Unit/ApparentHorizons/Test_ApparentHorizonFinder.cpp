// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <pup.h>
#include <random>
#include <vector>

#include "ApparentHorizons/ComputeItems.hpp"  // IWYU pragma: keep
#include "ApparentHorizons/FastFlow.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Informer/Tags.hpp"  // IWYU pragma: keep
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolationTargetApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor
/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace StrahlkorperTags {
template <typename Frame>
struct Radius;
template <typename Frame>
struct Strahlkorper;
}  // namespace StrahlkorperTags
namespace Tags {
template <class TagList>
struct Variables;
template <typename Tag, typename Dim, typename Frame, typename>
struct deriv;
}  // namespace Tags
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
/// \endcond

namespace {

// Counter to ensure that this function is called
size_t test_schwarzschild_horizon_called = 0;
struct TestSchwarzschildHorizon {
  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const typename Metavariables::temporal_id::
                        type& /*temporal_id*/) noexcept {
    const auto& horizon_radius =
        get<StrahlkorperTags::Radius<Frame::Inertial>>(box);
    const auto expected_radius =
        make_with_value<DataVector>(horizon_radius, 2.0);
    // We don't choose many grid points (for speed of test), so we
    // don't get the horizon radius extremely accurately.
    Approx custom_approx = Approx::custom().epsilon(1.e-2).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(horizon_radius, expected_radius,
                                 custom_approx);

    // Test that InverseSpatialMetric can be retrieved from the
    // DataBox and that its number of grid points is the same
    // as that of the strahlkorper.
    const auto& strahlkorper =
        get<StrahlkorperTags::Strahlkorper<Frame::Inertial>>(box);
    const auto& inv_metric =
        get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial>>(box);
    CHECK(strahlkorper.ylm_spherepack().physical_size() ==
          get<0,0>(inv_metric).size());

    ++test_schwarzschild_horizon_called;
  }
};

// Counter to ensure that this function is called
size_t test_kerr_horizon_called = 0;
struct TestKerrHorizon {
  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const typename Metavariables::temporal_id::
                        type& /*temporal_id*/) noexcept {
    const auto& strahlkorper =
        get<StrahlkorperTags::Strahlkorper<Frame::Inertial>>(box);
    // Test actual horizon radius against analytic value at the same
    // theta,phi points.
    const auto expected_radius = gr::Solutions::kerr_horizon_radius(
        strahlkorper.ylm_spherepack().theta_phi_points(), 1.1,
        {{0.12, 0.23, 0.45}});
    const auto& horizon_radius =
        get<StrahlkorperTags::Radius<Frame::Inertial>>(box);
    // The accuracy is not great because I use only a few grid points
    // to speed up the test.
    Approx custom_approx = Approx::custom().epsilon(1.e-3).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(horizon_radius, get(expected_radius),
                                 custom_approx);

    // Test that InverseSpatialMetric can be retrieved from the
    // DataBox and that its number of grid points is the same
    // as that of the strahlkorper.
    const auto& inv_metric =
        get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial>>(box);
    CHECK(strahlkorper.ylm_spherepack().physical_size() ==
          get<0,0>(inv_metric).size());

    ++test_kerr_horizon_called;
  }
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = Parallel::get_const_global_cache_tags<
      tmpl::list<typename InterpolationTargetTag::compute_target_points,
                 typename InterpolationTargetTag::post_interpolation_callback>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>>>>;
  using add_options_to_databox =
      typename intrp::Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template AddOptionsToDataBox<Metavariables>;

  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
};

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<intrp::Actions::InitializeInterpolator>>>;

  using component_being_mocked = intrp::Interpolator<Metavariables>;
};

template <typename PostHorizonFindCallback>
struct MockMetavariables {
  struct AhA : db::SimpleTag {
    using compute_items_on_source = tmpl::list<
        ah::Tags::InverseSpatialMetricCompute<3, Frame::Inertial>,
        ah::Tags::ExtrinsicCurvatureCompute<3, Frame::Inertial>,
        ah::Tags::SpatialChristoffelSecondKindCompute<3, Frame::Inertial>>;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::InverseSpatialMetric<3, Frame::Inertial>,
                   gr::Tags::ExtrinsicCurvature<3, Frame::Inertial>,
                   gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        intrp::Actions::ApparentHorizon<AhA, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<AhA>;
    using post_horizon_find_callback = PostHorizonFindCallback;
    // This `type` is so this tag can be used to read options.
    using type = typename compute_target_points::options_type;
    using container_tag = AhA;
    static std::string name() noexcept { return "AhA"; }
  };
  using interpolator_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<3, Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>;
  using interpolation_target_tags = tmpl::list<AhA>;
  using temporal_id  = ::Tags::TimeId;
  using domain_frame = Frame::Inertial;
  static constexpr size_t domain_dim = 3;
  using component_list =
      tmpl::list<mock_interpolation_target<MockMetavariables, AhA>,
                 mock_interpolator<MockMetavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

  enum class Phase { Initialization, Registration, Testing, Exit };
};

template <typename PostHorizonFindCallback>
void test_apparent_horizon(const gsl::not_null<size_t*> test_horizon_called,
                           const size_t l_max,
                           const size_t grid_points_each_dimension,
                           const double mass,
                           const std::array<double, 3>& dimensionless_spin) {
  using metavars = MockMetavariables<PostHorizonFindCallback>;
  using interp_component = mock_interpolator<metavars>;
  using target_component =
      mock_interpolation_target<metavars, typename metavars::AhA>;

  // The mocking framework requires an array index to be passed to
  // many functions, even if the components are singletons.
  constexpr size_t fake_array_index = 0_st;

  // Options for all InterpolationTargets.
  // The initial guess for the horizon search is a sphere of radius 2.8M.
  intrp::OptionHolders::ApparentHorizon<Frame::Inertial> apparent_horizon_opts(
      Strahlkorper<Frame::Inertial>{l_max, 2.8, {{0.0, 0.0, 0.0}}}, FastFlow{},
      Verbosity::Verbose);

  tuples::TaggedTuple<typename metavars::AhA> tuple_of_opts(
      std::move(apparent_horizon_opts));

  ActionTesting::MockRuntimeSystem<metavars> runner{std::move(tuple_of_opts)};

  // The test finds an apparent horizon for a Schwarzschild or Kerr
  // metric with M=1.  We choose a spherical shell domain extending
  // from radius 1.9M to 2.9M; this ensures the horizon is
  // inside the domain, and it gives a narrow domain so that we don't
  // need a large number of grid points to resolve the horizon (which
  // would make the test slower).
  const auto domain_creator = domain::creators::Shell<Frame::Inertial>(
      1.9, 2.9, 1, {{grid_points_each_dimension, grid_points_each_dimension}},
      false);

  runner.set_phase(metavars::Phase::Initialization);
  ActionTesting::emplace_component<interp_component>(&runner, 0);
  ActionTesting::next_action<interp_component>(make_not_null(&runner), 0);
  ActionTesting::emplace_component<target_component>(
      &runner, 0, domain_creator.create_domain());
  ActionTesting::next_action<target_component>(make_not_null(&runner), 0);
  runner.set_phase(metavars::Phase::Registration);

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
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::RegisterElement>(
        make_not_null(&runner), fake_array_index);
  }
  runner.set_phase(metavars::Phase::Testing);

  // Tell the InterpolationTargets that we want to interpolate at
  // temporal_id.
  ActionTesting::simple_action<
      target_component, intrp::Actions::AddTemporalIdsToInterpolationTarget<
                            typename metavars::AhA>>(
      make_not_null(&runner), 0, std::vector<TimeId>{temporal_id});

  // Create volume data and send it to the interpolator.
  for (const auto& element_id : element_ids) {
    const auto& block = domain.blocks()[element_id.block_id()];
    ::Mesh<3> mesh{domain_creator.initial_extents()[element_id.block_id()],
                   Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    ElementMap<3, Frame::Inertial> map{element_id,
                                       block.coordinate_map().get_clone()};
    const auto inertial_coords = map(logical_coordinates(mesh));

    // Compute psi, pi, phi for KerrSchild.
    gr::Solutions::KerrSchild solution(mass, dimensionless_spin,
                                       {{0.0, 0.0, 0.0}});
    const auto solution_vars = solution.variables(
        inertial_coords, 0.0, gr::Solutions::KerrSchild::tags<DataVector>{});
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
    const auto& dt_lapse =
        get<Tags::dt<gr::Tags::Lapse<DataVector>>>(solution_vars);
    const auto& d_lapse =
        get<gr::Solutions::KerrSchild::DerivLapse<DataVector>>(solution_vars);
    const auto& shift =
        get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(solution_vars);
    const auto& d_shift =
        get<gr::Solutions::KerrSchild::DerivShift<DataVector>>(solution_vars);
    const auto& dt_shift =
        get<Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataVector>>>(
            solution_vars);
    const auto& g =
        get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
            solution_vars);
    const auto& dt_g =
        get<Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
            solution_vars);
    const auto& d_g =
        get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(
            solution_vars);

    // Fill output variables with solution.
    db::item_type<
        ::Tags::Variables<typename metavars::interpolator_source_vars>>
        output_vars(mesh.number_of_grid_points());
    get<::gr::Tags::SpacetimeMetric<3, Frame::Inertial>>(output_vars) =
        gr::spacetime_metric(lapse, shift, g);
    get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(output_vars) =
        GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift, g, d_g);
    get<::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>>(output_vars) =
        GeneralizedHarmonic::pi(
            lapse, dt_lapse, shift, dt_shift, g, dt_g,
            get<::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>>(
                output_vars));

    // Call the InterpolatorReceiveVolumeData action on each element_id.
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::InterpolatorReceiveVolumeData>(
        make_not_null(&runner), 0, temporal_id, element_id, mesh,
        std::move(output_vars));
  }

  // Invoke remaining actions in random order.
  MAKE_GENERATOR(generator);
  auto index_map = ActionTesting::indices_of_components_with_queued_actions<
      typename metavars::component_list>(make_not_null(&runner),
                                         fake_array_index);
  while (not index_map.empty()) {
    ActionTesting::invoke_random_queued_action<
        typename metavars::component_list>(make_not_null(&runner),
                                           make_not_null(&generator), index_map,
                                           fake_array_index);
    index_map = ActionTesting::indices_of_components_with_queued_actions<
        typename metavars::component_list>(make_not_null(&runner),
                                           fake_array_index);
  }

  // Make sure function was called.
  CHECK(*test_horizon_called == 1);
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.ApparentHorizonFinder",
                  "[Unit]") {
  test_apparent_horizon<TestSchwarzschildHorizon>(
      &test_schwarzschild_horizon_called, 4, 5, 1.0, {{0.0, 0.0, 0.0}});
  test_apparent_horizon<TestKerrHorizon>(&test_kerr_horizon_called, 8, 5, 1.1,
                                         {{0.12, 0.23, 0.45}});
}
}  // namespace
