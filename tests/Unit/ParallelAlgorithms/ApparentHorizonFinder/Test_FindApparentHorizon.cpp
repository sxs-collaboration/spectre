// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "NumericalAlgorithms/Strahlkorper/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Callbacks/FailedHorizonFind.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Component.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeVarsToInterpolateToTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Factory.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Residual.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Shape.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Destination.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Initialization.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/HorizonMetavars.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
size_t callback_failure_count = 0;                                  // NOLINT
FastFlow::Status callback_failure_mode = FastFlow::Status::AbsTol;  // NOLINT
template <typename HorizonMetavars, size_t Index>
struct TestHorizonFindFailureCallback
    : tt::ConformsTo<ah::protocols::Callback> {
 private:
  using Fr = typename HorizonMetavars::frame;

 public:
  template <typename DbTags, typename Metavariables>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const FastFlow::Status failure_reason) {
    // Ignore error so we just increment the counter
    ah::callbacks::FailedHorizonFind<HorizonMetavars, true>::apply(
        box, cache, failure_reason);
    ++callback_failure_count;
    callback_failure_mode = failure_reason;
  }
};

size_t callback_count = 0;                   // NOLINT
std::vector<size_t> ah_found_resolutions{};  // NOLINT
template <typename HorizonMetavars, size_t Index>
struct TestHorizonFindCallback : tt::ConformsTo<ah::protocols::Callback> {
 private:
  using Fr = typename HorizonMetavars::frame;

 public:
  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const FastFlow::Status /*status*/) {
    ++callback_count;

    const auto& strahlkorper = get<ylm::Tags::Strahlkorper<Fr>>(box);
    // Test that InverseSpatialMetric can be retrieved from the
    // DataBox and that its number of grid points is the same
    // as that of the strahlkorper.
    const auto& inv_metric =
        get<gr::Tags::InverseSpatialMetric<DataVector, 3, Fr>>(box);
    CHECK(strahlkorper.ylm_spherepack().physical_size() ==
          get<0, 0>(inv_metric).size());

    const auto& current_resolution_l = strahlkorper.l_max();
    ah_found_resolutions.push_back(current_resolution_l);
  }
};

template <typename Fr, ah::Destination Dest>
struct HorizonMetavars : tt::ConformsTo<ah::protocols::HorizonMetavars> {
  using time_tag = ::Tags::TimeAndPrevious<0>;
  using frame = Fr;

  using horizon_find_callbacks =
      tmpl::list<TestHorizonFindCallback<HorizonMetavars, 0>,
                 TestHorizonFindCallback<HorizonMetavars, 1>>;
  using horizon_find_failure_callbacks =
      tmpl::list<TestHorizonFindFailureCallback<HorizonMetavars, 0>,
                 TestHorizonFindFailureCallback<HorizonMetavars, 1>>;

  using compute_tags_on_element = tmpl::list<>;

  static constexpr ah::Destination destination = Dest;

  static std::string name() { return "TestingHorizonMetavars"; }
};

template <typename Metavariables, typename HorizonMetavars>
struct MockComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using component_being_mocked = ah::Component<Metavariables, HorizonMetavars>;

  using frame = typename HorizonMetavars::frame;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<Initialization::Actions::InitializeItems<
          ah::Initialize<HorizonMetavars>>>>>;
};

template <typename Fr, ah::Destination Dest>
struct MockMetavariables {
  using component_list =
      tmpl::list<MockComponent<MockMetavariables, HorizonMetavars<Fr, Dest>>>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<3>,
                 ah::Tags::ApparentHorizonOptions<HorizonMetavars<Fr, Dest>>,
                 ah::Tags::LMax, ah::Tags::BlocksForHorizonFind>;
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<ah::Criterion, ah::Criteria::standard_criteria>>;
  };
};

template <typename Callbacks>
struct check_callbacks;

template <typename... Callbacks>
struct check_callbacks<tmpl::list<Callbacks...>>
    : std::integral_constant<bool,
                             tmpl2::flat_all_v<tt::assert_conforms_to_v<
                                 Callbacks, ah::protocols::Callback>...>> {};

template <typename Callbacks>
constexpr bool check_callbacks_v = check_callbacks<Callbacks>::value;

template <typename Fr, ah::Destination Dest = ah::Destination::ControlSystem,
          bool MakeHorizonFinderFailOnPurpose = false>
void test_apparent_horizon(
    const size_t l_max, const size_t grid_points_each_dimension,
    const double mass, const std::array<double, 3>& dimensionless_spin,
    const bool is_time_dependent,
    const std::optional<std::string>& dependency = std::nullopt,
    const size_t max_its = 100_st,
    std::vector<std::unique_ptr<ah::Criterion>> criteria = {}) {
  using metavars = MockMetavariables<Fr, Dest>;
  using horizon_metavars = HorizonMetavars<Fr, Dest>;
  using component = MockComponent<metavars, horizon_metavars>;

  // Assert the protocols
  static_assert(tt::assert_conforms_to_v<horizon_metavars,
                                         ah::protocols::HorizonMetavars>);
  static_assert(
      check_callbacks_v<typename horizon_metavars::horizon_find_callbacks>);
  static_assert(check_callbacks_v<
                typename horizon_metavars::horizon_find_failure_callbacks>);

  // The initial guess for the horizon search is a sphere of radius 2.8M unless
  // we are in the Distorted frame, then we pick 2.2 so it's within the blocks
  // that actually have a distorted frame
  ah::HorizonOptions<Fr> apparent_horizon_opts(
      std::move(criteria),
      ylm::Strahlkorper<Fr>{l_max,
                            std::is_same_v<Fr, ::Frame::Distorted> ? 2.2 : 2.8,
                            {{0.0, 0.0, 0.0}}},
      FastFlow{FastFlow::FlowType::Fast, 1.0, 0.5, 1.e-12, 1.e-2, 1.2, 5,
               max_its},
      Verbosity::Verbose, 3_st, std::nullopt);

  std::unordered_map<std::string, std::unordered_set<std::string>>
      blocks_for_interpolation{};

  // The test finds an apparent horizon for a Schwarzschild or Kerr metric with
  // M=1.  We choose a spherical shell domain extending from radius 1.9M to
  // 2.9M; this ensures the horizon is inside the domain, and it gives a narrow
  // domain so that we don't need a large number of grid points to resolve the
  // horizon (which would make the test slower).
  std::unique_ptr<DomainCreator<3>> domain_creator = std::make_unique<
      domain::creators::Sphere>(
      1.9, 2.9, domain::creators::Sphere::Excision{nullptr}, 1_st,
      grid_points_each_dimension, true, std::nullopt, std::vector<double>{2.4},
      domain::CoordinateMaps::Distribution::Linear, ShellWedges::All,
      is_time_dependent
          ? std::optional{domain::creators::sphere::TimeDependentMapOptions{
                0.0,
                domain::creators::time_dependent_options::ShapeMapOptions<
                    false, ::domain::ObjectLabel::None>{
                    l_max,
                    domain::creators::time_dependent_options::
                        KerrSchildFromBoyerLindquist{mass, dimensionless_spin}},
                std::nullopt, std::nullopt,
                domain::creators::time_dependent_options::TranslationMapOptions<
                    3>{std::array{std::array{0.0, 0.0, 0.0},
                                  std::array{0.0, 0.01, 0.0},
                                  std::array{0.0, 0.0, 0.0}}},
                true, std::nullopt}}
          : std::nullopt);

  {
    const Domain<3> domain_for_block_selection =
        domain_creator->create_domain();
    std::unordered_set<std::string> blocks_to_use{};
    for (const auto& block : domain_for_block_selection.blocks()) {
      if constexpr (std::is_same_v<Fr, ::Frame::Distorted>) {
        if (not block.has_distorted_frame()) {
          continue;
        }
      }
      blocks_to_use.insert(block.name());
    }
    blocks_for_interpolation.emplace("TestingHorizonMetavars",
                                     std::move(blocks_to_use));
  }

  const size_t max_resolution_and_output_l =
      std::max(l_max, static_cast<size_t>(12));

  ActionTesting::MockRuntimeSystem<metavars> runner{
      {domain_creator->create_domain(), std::move(apparent_horizon_opts),
       max_resolution_and_output_l, blocks_for_interpolation},
      {domain_creator->functions_of_time(),
       ah::Storage::LockedPreviousSurface<Fr>{}}};

  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_array_component<component>(
      &runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, 0_st);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Register);

  // Find horizon at three times.  The horizon find at the second time will use
  // the result from the first time as an initial guess.  The horizon find at
  // the third time will use an initial guess that was linearly extrapolated
  // from the first two horizon finds. For the time-independent case, the volume
  // data will not change between horizon finds, so the second and third horizon
  // finds will take zero iterations.  Having three times tests some logic in
  // the interpolator.
  const std::vector<LinkedMessageId<double>> times{
      {12.0 / 15.0, std::nullopt},
      {13.0 / 15.0, {12.0 / 15.0}},
      {14.0 / 15.0, {13.0 / 15.0}}};

  // Create element_ids.
  std::vector<ElementId<3>> element_ids{};
  const Domain<3> domain = domain_creator->create_domain();
  const auto& blocks_to_use =
      blocks_for_interpolation.at("TestingHorizonMetavars");
  const domain::FunctionsOfTimeMap functions_of_time =
      domain_creator->functions_of_time();
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator->initial_refinement_levels()[block.id()];
    auto elem_ids = initial_element_ids(block.id(), initial_ref_levs);
    element_ids.insert(element_ids.end(), elem_ids.begin(), elem_ids.end());
  }

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Center of the analytic solution.
  const auto analytic_solution_center = []() -> std::array<double, 3> {
    if constexpr (MakeHorizonFinderFailOnPurpose) {
      // Make the analytic solution off-center on purpose, so that the domain
      // only partially contains the horizon and therefore the interpolation
      // fails.
      return {0.5, 0.0, 0.0};
    }
    return {0.0, 0.0, 0.0};
  }();

  // Create volume data and send it to the interpolator, for each time.
  for (const auto& time : times) {
    for (const auto& element_id : element_ids) {
      const auto& block = domain.blocks()[element_id.block_id()];
      // Only send volume data for blocks in blocks_to_use
      if (blocks_to_use.find(block.name()) == blocks_to_use.end()) {
        continue;
      }
      const ::Mesh<3> mesh{
          domain_creator->initial_extents()[element_id.block_id()],
          Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};

      tnsr::I<DataVector, 3, ::Frame::Inertial> inertial_mesh_coords{};
      InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
          inv_jacobian_logical_to_inertial{mesh.number_of_grid_points(), 0.0};
      const auto logical_coords = logical_coordinates(mesh);
      if (domain.is_time_dependent()) {
        const ElementMap<3, Frame::Grid> map_logical_to_grid{
            element_id, block.moving_mesh_logical_to_grid_map().get_clone()};

        inertial_mesh_coords = block.moving_mesh_grid_to_inertial_map()(
            map_logical_to_grid(logical_coords), time.id, functions_of_time);

        const auto inv_jacobian_logical_to_grid =
            map_logical_to_grid.inv_jacobian(logical_coords);
        const auto inv_jacobian_grid_to_inertial =
            block.moving_mesh_grid_to_inertial_map().inv_jacobian(
                map_logical_to_grid(logical_coords), time.id,
                functions_of_time);

        inv_jacobian_logical_to_inertial = tenex::evaluate<ti::I, ti::j>(
            inv_jacobian_logical_to_grid(ti::I, ti::k) *
            inv_jacobian_grid_to_inertial(ti::K, ti::j));
      } else {
        // Accounts for Grid vs Inertial
        const ElementMap<3, ::Frame::Inertial> map{
            element_id, block.stationary_map().get_clone()};
        inertial_mesh_coords = map(logical_coords);

        inv_jacobian_logical_to_inertial = map.inv_jacobian(logical_coords);
      }

      // Compute g, pi, phi for KerrSchild.
      // Horizon is always at 0,0,0 in analytic_solution_coordinates.
      const gr::Solutions::KerrSchild solution(mass, dimensionless_spin,
                                               analytic_solution_center);
      const auto solution_vars = solution.variables(
          inertial_mesh_coords, 0.0,
          typename gr::Solutions::KerrSchild::tags<DataVector,
                                                   ::Frame::Inertial>{});

      // Fill output variables with solution.
      Variables<ah::source_vars<3>> source_vars(mesh.number_of_grid_points());

      const auto& lapse = get<gr::Tags::Lapse<DataVector>>(solution_vars);
      const auto& dt_lapse =
          get<Tags::dt<gr::Tags::Lapse<DataVector>>>(solution_vars);
      const auto& d_lapse = get<typename gr::Solutions::KerrSchild ::DerivLapse<
          DataVector, ::Frame::Inertial>>(solution_vars);
      const auto& shift = get<gr::Tags::Shift<DataVector, 3>>(solution_vars);
      const auto& d_shift = get<typename gr::Solutions::KerrSchild ::DerivShift<
          DataVector, ::Frame::Inertial>>(solution_vars);
      const auto& dt_shift =
          get<Tags::dt<gr::Tags::Shift<DataVector, 3>>>(solution_vars);
      const auto& g =
          get<gr::Tags::SpatialMetric<DataVector, 3>>(solution_vars);
      const auto& dt_g =
          get<Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(solution_vars);
      const auto& d_g =
          get<typename gr::Solutions::KerrSchild ::DerivSpatialMetric<
              DataVector, ::Frame::Inertial>>(solution_vars);

      get<::gr::Tags::SpacetimeMetric<DataVector, 3>>(source_vars) =
          gr::spacetime_metric(lapse, shift, g);
      get<::gh::Tags::Phi<DataVector, 3>>(source_vars) =
          gh::phi(lapse, d_lapse, shift, d_shift, g, d_g);
      get<::gh::Tags::Pi<DataVector, 3>>(source_vars) =
          gh::pi(lapse, dt_lapse, shift, dt_shift, g, dt_g,
                 get<::gh::Tags::Phi<DataVector, 3>>(source_vars));

      // Need to compute numerical deriv of Phi.
      get<Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                      Frame::Inertial>>(source_vars) =
          partial_derivative(get<::gh::Tags::Phi<DataVector, 3>>(source_vars),
                             mesh, inv_jacobian_logical_to_inertial);

      // TO-DO: make target_vars from the source_vars in the correct frame
      Variables<ah::vars_to_interpolate_to_target<3, Fr>> target_vars{
          get(lapse).size()};
      ah::compute_vars_to_interpolate_to_target(
          make_not_null(&target_vars),
          get<::gr::Tags::SpacetimeMetric<DataVector, 3>>(source_vars),
          get<::gh::Tags::Pi<DataVector, 3>>(source_vars),
          get<::gh::Tags::Phi<DataVector, 3>>(source_vars),
          get<Tags::deriv<::gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                          Frame::Inertial>>(source_vars),
          time, domain, mesh, element_id, functions_of_time);

      // Queue the action so we can invoke in a random order below
      ActionTesting::queue_simple_action<
          component, ah::FindApparentHorizon<horizon_metavars>>(
          make_not_null(&runner), 0, time, element_id, mesh, target_vars,
          dependency);
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
}

// [[TimeOut, 60]]
SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.FindApparentHorizon",
                  "[ApparentHorizonFinder][Unit]") {
  domain::creators::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  register_factory_classes_with_charm<
      MockMetavariables<Frame::Inertial, ah::Destination::ControlSystem>>();

  const std::optional<std::string> dependency{"FakeDependency"};

  // Time-independent tests.
  callback_failure_count = 0;
  callback_count = 0;
  test_apparent_horizon<Frame::Inertial>(3, 3, 1.0, {{0.0, 0.0, 0.0}}, false,
                                         dependency);
  test_apparent_horizon<Frame::Grid>(3, 4, 1.1, {{0.2, 0.1, -0.4}}, false);
  test_apparent_horizon<Frame::Inertial, ah::Destination::Observation>(
      3, 4, 1.1, {{0.2, 0.1, -0.4}}, false);
  CHECK(callback_count == 18);
  CHECK(callback_failure_count == 0);
  CHECK(ah_found_resolutions == std::vector<size_t>(18, 3));
  callback_count = 0;
  ah_found_resolutions.clear();

  // Time-dependent tests.
  test_apparent_horizon<Frame::Inertial>(3, 5, 1.1, {{0.2, 0.1, -0.4}}, true);
  test_apparent_horizon<Frame::Distorted>(3, 4, 1.1, {{0.2, 0.1, -0.4}}, true,
                                          dependency);
  test_apparent_horizon<Frame::Distorted, ah::Destination::Observation>(
      3, 4, 1.1, {{0.2, 0.1, -0.4}}, true);
  test_apparent_horizon<Frame::Grid>(3, 3, 1.0, {{0.0, 0.0, 0.0}}, true);
  CHECK(callback_count == 24);
  CHECK(callback_failure_count == 0);
  CHECK(ah_found_resolutions == std::vector<size_t>(24, 3));
  callback_count = 0;
  ah_found_resolutions.clear();

  // Adaptivity tests
  // First, choose strict critera so the resolution increases
  // by one each time.
  const ah::Criteria::Residual residual_criterion{1.e-7, 9.e-5, 4};
  const ah::Criteria::Shape shape_criterion{1.e-7, 9.e-5, 20, 4};
  std::vector<std::unique_ptr<ah::Criterion>> criteria{};
  criteria.emplace_back(
      std::make_unique<ah::Criteria::Residual>(residual_criterion));
  criteria.emplace_back(std::make_unique<ah::Criteria::Shape>(shape_criterion));

  test_apparent_horizon<Frame::Inertial>(3, 3, 1.0, {{0.2, 0.2, 0.2}}, false,
                                         dependency, 100_st,
                                         std::move(criteria));
  CHECK(callback_count == 6);
  CHECK(callback_failure_count == 0);
  CHECK(ah_found_resolutions == std::vector<size_t>(6, 8));

  callback_count = 0;
  ah_found_resolutions.clear();

  // Second, choose loose critera so the resolution increases
  // by one each time
  const ah::Criteria::Residual residual_criterion_loose{1.0e8, 1.0e12, 4};
  const ah::Criteria::Shape shape_criterion_loose{1.0e8, 1.0e12, 20, 4};
  criteria.clear();
  criteria.emplace_back(
      std::make_unique<ah::Criteria::Residual>(residual_criterion_loose));
  criteria.emplace_back(
      std::make_unique<ah::Criteria::Shape>(shape_criterion_loose));

  test_apparent_horizon<Frame::Inertial>(8, 8, 1.0, {{0.0, 0.0, 0.0}}, false,
                                         dependency, 100_st,
                                         std::move(criteria));
  CHECK(callback_count == 6);
  CHECK(callback_failure_count == 0);
  CHECK(ah_found_resolutions == std::vector<size_t>{8, 8, 7, 7, 6, 6});
  callback_count = 0;
  ah_found_resolutions.clear();

  // Failure tests
  test_apparent_horizon<Frame::Inertial, ah::Destination::ControlSystem, true>(
      3, 3, 1.0, {{0.0, 0.0, 0.0}}, true);
  CHECK(callback_count == 0);
  CHECK(callback_failure_count == 6);
  CHECK(callback_failure_mode == FastFlow::Status::InterpolationFailure);
  callback_failure_count = 0;

  test_apparent_horizon<Frame::Grid, ah::Destination::ControlSystem, true>(
      3, 3, 10.0, {{0.0, 0.0, 0.0}}, false);
  CHECK(callback_count == 0);
  CHECK(callback_failure_count == 6);
  CHECK(callback_failure_mode == FastFlow::Status::InterpolationFailure);
  callback_failure_count = 0;

  test_apparent_horizon<Frame::Inertial, ah::Destination::ControlSystem, true>(
      3, 3, 1.0, {{0.0, 0.0, 0.0}}, true, dependency, 1);
  CHECK(callback_count == 0);
  CHECK(callback_failure_count == 6);
  CHECK(callback_failure_mode == FastFlow::Status::MaxIts);
}
}  // namespace
