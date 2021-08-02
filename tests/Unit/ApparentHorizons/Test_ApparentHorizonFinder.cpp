// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <random>
#include <utility>
#include <vector>

#include "ApparentHorizons/ComputeHorizonVolumeQuantities.hpp"
#include "ApparentHorizons/ComputeHorizonVolumeQuantities.tpp"
#include "ApparentHorizons/ComputeItems.hpp"  // IWYU pragma: keep
#include "ApparentHorizons/FastFlow.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Logging/Tags.hpp"  // IWYU pragma: keep
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Interpolation/AddTimesToInterpolationTarget.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/Callbacks/ErrorOnFailedApparentHorizon.hpp"
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
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
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

namespace {

// This struct is here to generate a snippet for the dox.
// Otherwise, we would just call ErrorOnFailedApparentHorizon directly.
struct TestHorizonFindFailureCallback {
  // [horizon_find_failure_callback_example]
  template <typename InterpolationTargetTag, typename DbTags,
            typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const double time,
                    const FastFlow::Status failure_reason) noexcept {
    // [horizon_find_failure_callback_example]
    intrp::callbacks::ErrorOnFailedApparentHorizon::template apply<
        InterpolationTargetTag>(box, cache, time, failure_reason);
  }
};

// Counter to ensure that this function is called
size_t test_schwarzschild_horizon_called = 0;
template <typename Frame>
struct TestSchwarzschildHorizon {
  using observation_types = tmpl::list<>;
  // [post_horizon_find_callback_example]
  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const double /*time*/) noexcept {
    // [post_horizon_find_callback_example]
    const auto& horizon_radius = get(get<StrahlkorperTags::Radius<Frame>>(box));
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
    const auto& strahlkorper = get<StrahlkorperTags::Strahlkorper<Frame>>(box);
    const auto& inv_metric = get<gr::Tags::InverseSpatialMetric<3, Frame>>(box);
    CHECK(strahlkorper.ylm_spherepack().physical_size() ==
          get<0, 0>(inv_metric).size());

    ++test_schwarzschild_horizon_called;
  }
};

// Counter to ensure that this function is called
size_t test_kerr_horizon_called = 0;
template <typename Frame>
struct TestKerrHorizon {
  using observation_types = tmpl::list<>;
  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const double /*time*/) noexcept {
    const auto& strahlkorper =
        get<StrahlkorperTags::Strahlkorper<Frame>>(box);
    // Test actual horizon radius against analytic value at the same
    // theta,phi points.
    const auto expected_radius = gr::Solutions::kerr_horizon_radius(
        strahlkorper.ylm_spherepack().theta_phi_points(), 1.1,
        {{0.12, 0.23, 0.45}});
    const auto& horizon_radius = get(get<StrahlkorperTags::Radius<Frame>>(box));
    // The accuracy is not great because I use only a few grid points
    // to speed up the test.
    Approx custom_approx = Approx::custom().epsilon(1.e-3).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(horizon_radius, get(expected_radius),
                                 custom_approx);

    // Test that InverseSpatialMetric can be retrieved from the
    // DataBox and that its number of grid points is the same
    // as that of the strahlkorper.
    const auto& inv_metric =
        get<gr::Tags::InverseSpatialMetric<3, Frame>>(box);
    CHECK(strahlkorper.ylm_spherepack().physical_size() ==
          get<0, 0>(inv_metric).size());

    ++test_kerr_horizon_called;
  }
};

template <typename Metavariables, typename InterpolationTargetTag>
struct mock_interpolation_target {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = size_t;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<tmpl::list<
          typename InterpolationTargetTag::compute_target_points,
          typename InterpolationTargetTag::post_interpolation_callback>>;
  using mutable_global_cache_tags =
      tmpl::conditional_t<metavariables::use_time_dependent_maps,
                          tmpl::list<domain::Tags::FunctionsOfTime>,
                          tmpl::list<>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<Actions::SetupDataBox,
                 intrp::Actions::InitializeInterpolationTarget<
                     Metavariables, InterpolationTargetTag>>>>;

  using component_being_mocked =
      intrp::InterpolationTarget<Metavariables, InterpolationTargetTag>;
};

template <typename Metavariables>
struct mock_interpolator {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<Actions::SetupDataBox,
                 intrp::Actions::InitializeInterpolator<
                     intrp::Tags::VolumeVarsInfo<Metavariables>,
                     intrp::Tags::InterpolatedVarsHolders<Metavariables>>>>>;

  using component_being_mocked = intrp::Interpolator<Metavariables>;
};

template <typename PostHorizonFindCallback, typename IsTimeDependent,
          typename TargetFrame>
struct MockMetavariables {
  static constexpr bool use_time_dependent_maps = IsTimeDependent::value;
  struct AhA {
    using compute_vars_to_interpolate = ah::ComputeHorizonVolumeQuantities;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::InverseSpatialMetric<3, TargetFrame>,
                   gr::Tags::ExtrinsicCurvature<3, TargetFrame>,
                   gr::Tags::SpatialChristoffelSecondKind<3, TargetFrame>>;
    using compute_items_on_target = tmpl::list<>;
    using compute_target_points =
        intrp::TargetPoints::ApparentHorizon<AhA, TargetFrame>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<AhA, TargetFrame>;
    using post_horizon_find_callback = PostHorizonFindCallback;
    using horizon_find_failure_callback = TestHorizonFindFailureCallback;
  };
  using interpolator_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>,
                 GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>;
  using interpolation_target_tags = tmpl::list<AhA>;
  static constexpr size_t volume_dim = 3;
  using component_list =
      tmpl::list<mock_interpolation_target<MockMetavariables, AhA>,
                 mock_interpolator<MockMetavariables>>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<3>>;

  enum class Phase { Initialization, Registration, Testing, Exit };
};

template <typename PostHorizonFindCallback, typename IsTimeDependent,
          typename Frame = ::Frame::Inertial,
          bool MakeHorizonFinderFailOnPurpose = false>
void test_apparent_horizon(const gsl::not_null<size_t*> test_horizon_called,
                           const size_t l_max,
                           const size_t grid_points_each_dimension,
                           const double mass,
                           const std::array<double, 3>& dimensionless_spin) {
  using metavars =
      MockMetavariables<PostHorizonFindCallback, IsTimeDependent, Frame>;
  using interp_component = mock_interpolator<metavars>;
  using target_component =
      mock_interpolation_target<metavars, typename metavars::AhA>;

  // Options for all InterpolationTargets.
  // The initial guess for the horizon search is a sphere of radius 2.8M.
  intrp::OptionHolders::ApparentHorizon<Frame> apparent_horizon_opts(
      Strahlkorper<Frame>{l_max, 2.8, {{0.0, 0.0, 0.0}}}, FastFlow{},
      Verbosity::Verbose);

  std::unique_ptr<DomainCreator<3>> domain_creator;
  std::unique_ptr<ActionTesting::MockRuntimeSystem<metavars>> runner_ptr{};

  // The test finds an apparent horizon for a Schwarzschild or Kerr
  // metric with M=1.  We choose a spherical shell domain extending
  // from radius 1.9M to 2.9M; this ensures the horizon is
  // inside the domain, and it gives a narrow domain so that we don't
  // need a large number of grid points to resolve the horizon (which
  // would make the test slower).
  if constexpr (IsTimeDependent::value) {
    const double expiration_time = 1.0;
    std::vector<double> radial_partitioning{};
    std::vector<domain::CoordinateMaps::Distribution> radial_distribution{
        domain::CoordinateMaps::Distribution::Linear};

    domain_creator = std::make_unique<domain::creators::Shell>(
        1.9, 2.9, 1,
        std::array<size_t, 2>{grid_points_each_dimension,
                              grid_points_each_dimension},
        false, 1.0, radial_partitioning, radial_distribution, ShellWedges::All,
        std::make_unique<
            domain::creators::time_dependence::UniformTranslation<3>>(
            0.0, expiration_time,
            std::array<double, 3>({{0.01, 0.02, 0.03}})));
    tuples::TaggedTuple<
        domain::Tags::Domain<3>,
        typename ::intrp::Tags::ApparentHorizon<typename metavars::AhA, Frame>>
        tuple_of_opts{std::move(domain_creator->create_domain()),
                      std::move(apparent_horizon_opts)};
    runner_ptr = std::make_unique<ActionTesting::MockRuntimeSystem<metavars>>(
        std::move(tuple_of_opts), domain_creator->functions_of_time(),
        std::vector<size_t>{3, 2});
  } else {
    domain_creator = std::make_unique<domain::creators::Shell>(
        1.9, 2.9, 1,
        std::array<size_t, 2>{grid_points_each_dimension,
                              grid_points_each_dimension},
        false);

    tuples::TaggedTuple<
        domain::Tags::Domain<3>,
        typename ::intrp::Tags::ApparentHorizon<typename metavars::AhA, Frame>>
        tuple_of_opts{std::move(domain_creator->create_domain()),
                      std::move(apparent_horizon_opts)};

    runner_ptr = std::make_unique<ActionTesting::MockRuntimeSystem<metavars>>(
        std::move(tuple_of_opts), tuples::TaggedTuple<>{},
        std::vector<size_t>{3, 2});
  }
  auto& runner = *runner_ptr;

  ActionTesting::set_phase(make_not_null(&runner),
                           metavars::Phase::Initialization);
  ActionTesting::emplace_group_component<interp_component>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t indx = 0; indx < 5; ++indx) {
      ActionTesting::next_action<interp_component>(make_not_null(&runner),
                                                   indx);
    }
  }
  ActionTesting::emplace_singleton_component<target_component>(
      &runner, ActionTesting::NodeId{1}, ActionTesting::LocalCoreId{1});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<target_component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           metavars::Phase::Registration);

  // Find horizon at two times.  The horizon find at the second
  // time will use the result from the first time as
  // an initial guess.  For the time-independent case, the volume data will
  // not change between horizon finds, so the second horizon find will take
  // zero iterations.
  // Having two times tests some logic in the interpolator.
  const std::vector<double> times{13.0 / 15.0, 14.0 / 15.0};

  // Create element_ids.
  std::vector<ElementId<3>> element_ids{};
  Domain<3> domain = domain_creator->create_domain();
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator->initial_refinement_levels()[block.id()];
    auto elem_ids = initial_element_ids(block.id(), initial_ref_levs);
    element_ids.insert(element_ids.end(), elem_ids.begin(), elem_ids.end());
  }

  // Tell the interpolator how many elements there are by registering
  // each one.  Normally intrp::Actions::RegisterElement is called by
  // RegisterElementWithInterpolator, and invoked on the ckLocalBranch
  // of the interpolator that is associated with each element
  // (i.e. the local core on each element).
  // Here we assign elements round-robin to the mock cores.
  // And for group components, the array_index is the global core index.
  const size_t num_cores = runner.num_global_cores();
  std::unordered_map<ElementId<3>, size_t> mock_core_for_each_element;
  size_t core = 0;
  for (const auto& element_id : element_ids) {
    mock_core_for_each_element.insert({element_id, core});
    ActionTesting::simple_action<interp_component,
                                 intrp::Actions::RegisterElement>(
        make_not_null(&runner), core);
    if (++core >= num_cores) {
      core = 0;
    }
  }
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  // Tell the InterpolationTargets that we want to interpolate at
  // two times.
  ActionTesting::simple_action<
      target_component,
      intrp::Actions::AddTimesToInterpolationTarget<typename metavars::AhA>>(
      make_not_null(&runner), 0, times);

  // Center of the analytic solution.
  const auto analytic_solution_center = []() noexcept -> std::array<double, 3> {
    if constexpr (MakeHorizonFinderFailOnPurpose) {
      // Make the analytic solution off-center on purpose, so that
      // the domain only partially contains the horizon and therefore
      // the interpolation fails.
      return {0.5, 0.0, 0.0};
    }
    return {0.0, 0.0, 0.0};
  }();

  // Create volume data and send it to the interpolator, for each time.
  for (const auto& time : times) {
    for (const auto& element_id : element_ids) {
      const auto& block = domain.blocks()[element_id.block_id()];
      ::Mesh<3> mesh{domain_creator->initial_extents()[element_id.block_id()],
                     Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

      tnsr::I<DataVector, 3, Frame> analytic_solution_coords{};
      if constexpr (std::is_same_v<Frame, ::Frame::Grid> and
                    IsTimeDependent::value) {
        ElementMap<3, ::Frame::Grid> map_logical_to_grid{
            element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
        analytic_solution_coords =
            map_logical_to_grid(logical_coordinates(mesh));
      } else if constexpr (IsTimeDependent::value) {
        // We don't have an Element ParallelComponent in this test, so
        // get the cache from the target component.
        ElementMap<3, ::Frame::Grid> map_logical_to_grid{
            element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
        const auto& cache =
            ActionTesting::cache<target_component>(runner, 0_st);
        const auto& functions_of_time =
            get<domain::Tags::FunctionsOfTime>(cache);
        analytic_solution_coords = block.moving_mesh_grid_to_inertial_map()(
            map_logical_to_grid(logical_coordinates(mesh)), time,
            functions_of_time);
      } else {
        // Time-independent
        ElementMap<3, ::Frame::Inertial> map{
            element_id, block.stationary_map().get_clone()};
        analytic_solution_coords = map(logical_coordinates(mesh));
      }

      // Compute psi, pi, phi for KerrSchild.
      // Horizon is always at 0,0,0 in analytic_solution_coordinates.
      gr::Solutions::KerrSchild solution(mass, dimensionless_spin,
                                         analytic_solution_center);
      const auto solution_vars = solution.variables(
          analytic_solution_coords, 0.0,
          typename gr::Solutions::KerrSchild::tags<DataVector, Frame>{});

      // Fill output variables with solution.
      typename ::Tags::Variables<typename metavars::interpolator_source_vars>::
          type output_vars(mesh.number_of_grid_points());

      if constexpr (std::is_same_v<Frame, ::Frame::Inertial>) {
        // Easy case: Grid and Inertial frame are the same

        const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
        const auto& dt_lapse =
            get<Tags::dt<gr::Tags::Lapse<DataVector>>>(solution_vars);
        const auto& d_lapse = get<
            typename gr::Solutions::KerrSchild ::DerivLapse<DataVector, Frame>>(
            solution_vars);
        const auto& shift =
            get<gr::Tags::Shift<3, Frame, DataVector>>(solution_vars);
        const auto& d_shift = get<
            typename gr::Solutions::KerrSchild ::DerivShift<DataVector, Frame>>(
            solution_vars);
        const auto& dt_shift =
            get<Tags::dt<gr::Tags::Shift<3, Frame, DataVector>>>(solution_vars);
        const auto& g =
            get<gr::Tags::SpatialMetric<3, Frame, DataVector>>(solution_vars);
        const auto& dt_g =
            get<Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataVector>>>(
                solution_vars);
        const auto& d_g =
            get<typename gr::Solutions::KerrSchild ::DerivSpatialMetric<
                DataVector, Frame>>(solution_vars);

        get<::gr::Tags::SpacetimeMetric<3, ::Frame::Inertial>>(output_vars) =
            gr::spacetime_metric(lapse, shift, g);
        get<::GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
            output_vars) =
            GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift, g, d_g);
        get<::GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(
            output_vars) =
            GeneralizedHarmonic::pi(
                lapse, dt_lapse, shift, dt_shift, g, dt_g,
                get<::GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
                    output_vars));
      } else {
        // Frame is not Inertial, so need to transform tensors to
        // Inertial frame, since InterpolatorReceiveVolumeData always gets
        // its volume data in the Inertial frame.

        // The difficult parts are Pi and Phi, which are not tensors,
        // so they are not so easy to transform because we do not have
        // Hessians.
        //
        // So what we do is compute only the 3-metric, lapse, shift,
        // and extrinsic curvature, transform them (because they are
        // 3-tensors), and compute the other components by numerical
        // differentiation.
        const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
        const auto& shift =
            get<gr::Tags::Shift<3, Frame, DataVector>>(solution_vars);
        const auto& g =
            get<gr::Tags::SpatialMetric<3, Frame, DataVector>>(solution_vars);
        const auto& K = get<gr::Tags::ExtrinsicCurvature<3, Frame, DataVector>>(
            solution_vars);

        auto& cache = ActionTesting::cache<target_component>(runner, 0_st);
        const auto& functions_of_time =
            get<domain::Tags::FunctionsOfTime>(cache);
        const auto coords_frame_velocity_jacobians =
            block.moving_mesh_grid_to_inertial_map()
                .coords_frame_velocity_jacobians(analytic_solution_coords, time,
                                                 functions_of_time);
        const auto& inv_jacobian = std::get<1>(coords_frame_velocity_jacobians);
        const auto& jacobian = std::get<2>(coords_frame_velocity_jacobians);
        const auto& frame_velocity =
            std::get<3>(coords_frame_velocity_jacobians);

        using inertial_metric_vars_tags = tmpl::list<
            gr::Tags::Lapse<DataVector>,
            gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
            gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>;
        Variables<inertial_metric_vars_tags> inertial_metric_vars(
            mesh.number_of_grid_points());

        auto& shift_inertial =
            get<::gr::Tags::Shift<3, ::Frame::Inertial>>(inertial_metric_vars);
        auto& lower_metric_inertial =
            get<::gr::Tags::SpatialMetric<3, ::Frame::Inertial>>(
                inertial_metric_vars);
        // Just copy lapse, since it doesn't transform. Need it for derivs.
        get<gr::Tags::Lapse<DataVector>>(inertial_metric_vars) = lapse;

        for (size_t k = 0; k < 3; ++k) {
          shift_inertial.get(k) = -frame_velocity.get(k);
          for (size_t j = 0; j < 3; ++j) {
            shift_inertial.get(k) += jacobian.get(k, j) * shift.get(j);
          }
        }

        auto transform =
            [&inv_jacobian](
                const gsl::not_null<tnsr::ii<DataVector, 3, ::Frame::Inertial>*>
                    u_inertial,
                const tnsr::ii<DataVector, 3, Frame>& u_frame) noexcept {
              for (size_t i = 0; i < 3; ++i) {
                for (size_t j = i; j < 3; ++j) {  // symmetry
                  u_inertial->get(i, j) = 0.0;
                  for (size_t k = 0; k < 3; ++k) {
                    for (size_t p = 0; p < 3; ++p) {
                      u_inertial->get(i, j) += inv_jacobian.get(k, i) *
                                               inv_jacobian.get(p, j) *
                                               u_frame.get(k, p);
                    }
                  }
                }
              }
            };

        transform(make_not_null(&lower_metric_inertial), g);
        tnsr::ii<DataVector, 3, ::Frame::Inertial> extrinsic_curvature_inertial(
            mesh.number_of_grid_points());
        transform(make_not_null(&extrinsic_curvature_inertial), K);

        // Take spatial derivatives
        ElementMap<3, ::Frame::Grid> map_logical_to_grid{
            element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
        const auto grid_deriv_inertial_metric_vars =
            partial_derivatives<inertial_metric_vars_tags>(
                inertial_metric_vars, mesh,
                map_logical_to_grid.inv_jacobian(logical_coordinates(mesh)));
        tnsr::ijj<DataVector, 3, ::Frame::Inertial> d_g(
            mesh.number_of_grid_points());
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = i; j < 3; ++j) {  // symmetry
            for (size_t k = 0; k < 3; ++k) {
              d_g.get(k, i, j) = 0.0;
              for (size_t l = 0; l < 3; ++l) {
                d_g.get(k, i, j) +=
                    inv_jacobian.get(l, k) *
                    get<Tags::deriv<gr::Tags::SpatialMetric<
                                        3, ::Frame::Inertial, DataVector>,
                                    tmpl::size_t<3>, ::Frame::Grid>>(
                        grid_deriv_inertial_metric_vars)
                        .get(l, i, j);
              }
            }
          }
        }
        tnsr::iJ<DataVector, 3, ::Frame::Inertial> d_shift(
            mesh.number_of_grid_points());
        for (size_t i = 0; i < 3; ++i) {
          for (size_t k = 0; k < 3; ++k) {
            d_shift.get(k, i) = 0.0;
            for (size_t l = 0; l < 3; ++l) {
              d_shift.get(k, i) +=
                  inv_jacobian.get(l, k) *
                  get<Tags::deriv<
                      gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
                      tmpl::size_t<3>, ::Frame::Grid>>(
                      grid_deriv_inertial_metric_vars)
                      .get(l, i);
            }
          }
        }
        tnsr::i<DataVector, 3, ::Frame::Inertial> d_lapse(
            mesh.number_of_grid_points());
        for (size_t k = 0; k < 3; ++k) {
          d_lapse.get(k) = 0.0;
          for (size_t l = 0; l < 3; ++l) {
            d_lapse.get(k) +=
                inv_jacobian.get(l, k) *
                get<Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                                ::Frame::Grid>>(grid_deriv_inertial_metric_vars)
                    .get(l);
          }
        }

        get<::gr::Tags::SpacetimeMetric<3, ::Frame::Inertial>>(output_vars) =
            gr::spacetime_metric(lapse, shift_inertial, lower_metric_inertial);
        get<::GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
            output_vars) =
            GeneralizedHarmonic::phi(lapse, d_lapse, shift_inertial, d_shift,
                                     lower_metric_inertial, d_g);
        // Compute Pi from extrinsic curvature and Phi.  Fill in zero
        // for zero components of Pi, since they won't be used at all
        // (can't fill in NaNs, because they will still be interpolated).
        const auto spacetime_normal_vector =
            gr::spacetime_normal_vector(lapse, shift_inertial);
        auto& Pi = get<::GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(
            output_vars);
        const auto& Phi =
            get<::GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
                output_vars);
        for (size_t i = 0; i < 3; ++i) {
          Pi.get(i + 1, 0) = 0.0;
          for (size_t j = i; j < 3; ++j) {  // symmetry
            Pi.get(i + 1, j + 1) =
                2.0 * extrinsic_curvature_inertial.get(i, j);
            for (size_t c = 0; c < 4; ++c) {
              Pi.get(i + 1, j + 1) -=
                  spacetime_normal_vector.get(c) *
                  (Phi.get(i , j + 1, c) + Phi.get(j , i + 1, c));
            }
          }
        }
        Pi.get(0, 0) = 0.0;
      }

      // Call the InterpolatorReceiveVolumeData action on each element_id.
      ActionTesting::simple_action<
          interp_component, intrp::Actions::InterpolatorReceiveVolumeData>(
          make_not_null(&runner), mock_core_for_each_element.at(element_id),
          time, element_id, mesh, output_vars);
    }
  }

  // Invoke remaining actions in random order.
  MAKE_GENERATOR(generator);
  auto array_indices_with_queued_simple_actions =
      ActionTesting::array_indices_with_queued_simple_actions<
          typename metavars::component_list>(make_not_null(&runner));
  while (ActionTesting::number_of_elements_with_queued_simple_actions<
             typename metavars::component_list>(
             array_indices_with_queued_simple_actions) > 0) {
    ActionTesting::invoke_random_queued_simple_action<
        typename metavars::component_list>(
        make_not_null(&runner), make_not_null(&generator),
        array_indices_with_queued_simple_actions);
    array_indices_with_queued_simple_actions =
        ActionTesting::array_indices_with_queued_simple_actions<
            typename metavars::component_list>(make_not_null(&runner));
  }

  // Make sure function was called twice.
  CHECK(*test_horizon_called == 2);
}

// This tests the entire AH finder including numerical interpolation.
// We increase the timeout because we want to test time-independent
// and time-dependent cases, and we want to test more than one frame.
// Already the resolution used for the tests is very low
// (lmax=3, num_pts_per_dim=3 to 7) and the error
// tolerance Approx::custom().epsilon() is pretty large (1e-2 and 1e-3).
// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Interpolator.ApparentHorizonFinder",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  // Time-independent tests.
  test_schwarzschild_horizon_called = 0;
  test_kerr_horizon_called = 0;
  test_apparent_horizon<TestSchwarzschildHorizon<Frame::Inertial>,
                        std::false_type>(&test_schwarzschild_horizon_called, 3,
                                         3, 1.0, {{0.0, 0.0, 0.0}});
  test_apparent_horizon<TestKerrHorizon<Frame::Inertial>, std::false_type>(
      &test_kerr_horizon_called, 3, 5, 1.1, {{0.12, 0.23, 0.45}});

  // Time-dependent tests.
  test_schwarzschild_horizon_called = 0;
  test_kerr_horizon_called = 0;
  test_apparent_horizon<TestSchwarzschildHorizon<Frame::Inertial>,
                        std::true_type>(&test_schwarzschild_horizon_called, 3,
                                        4, 1.0, {{0.0, 0.0, 0.0}});
  test_apparent_horizon<TestKerrHorizon<Frame::Inertial>, std::true_type>(
      &test_kerr_horizon_called, 3, 5, 1.1, {{0.12, 0.23, 0.45}});

  // Time-dependent tests in grid frame.
  test_schwarzschild_horizon_called = 0;
  test_kerr_horizon_called = 0;
  test_apparent_horizon<TestSchwarzschildHorizon<Frame::Grid>, std::true_type,
                        Frame::Grid>(&test_schwarzschild_horizon_called, 3, 6,
                                     1.0, {{0.0, 0.0, 0.0}});
  test_apparent_horizon<TestKerrHorizon<Frame::Grid>, std::true_type,
                        Frame::Grid>(&test_kerr_horizon_called, 3, 7, 1.1,
                                     {{0.12, 0.23, 0.45}});
}

// [[OutputRegex, Cannot interpolate onto surface]]
SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.Interpolator.ApparentHorizonFinder.Error",
    "[Unit]") {
  ERROR_TEST();
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();

  test_schwarzschild_horizon_called = 0;
  test_apparent_horizon<TestSchwarzschildHorizon<Frame::Inertial>,
                        std::true_type, Frame::Inertial, true>(
      &test_schwarzschild_horizon_called, 3, 4, 1.0, {{0.0, 0.0, 0.0}});
}
}  // namespace
