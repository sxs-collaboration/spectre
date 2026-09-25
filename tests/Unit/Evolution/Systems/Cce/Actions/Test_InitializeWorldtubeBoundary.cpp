// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RobinsonTrautman.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Initialize/CauchySecondOrder.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/Actions/WorldtubeBoundaryMocking.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
// Required when registering all SpanInterpolator subclasses with Charm++.
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/TimeSteppers/AdamsMoultonPc.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
template <class Metavariables>
struct AnalyticWorldtubeBoundary;
template <class Metavariables>
struct GhWorldtubeBoundary;
template <class Metavariables>
struct H5WorldtubeBoundary;
}  // namespace Cce
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags

namespace Cce {

namespace {

struct H5Metavariables {
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using component_list =
      tmpl::list<mock_worldtube_boundary<H5Metavariables,
                                         H5WorldtubeBoundary<H5Metavariables>>>;
};

// The shared mock does not carry the initial-data generator; the production
// `H5WorldtubeBoundary` does, and the initialization action reads the
// generator's Du(Dr(J)) interpolator out of it.
template <typename Metavariables>
struct mock_h5_worldtube_boundary_with_initialize_j
    : mock_worldtube_boundary<Metavariables,
                              H5WorldtubeBoundary<Metavariables>> {
  using const_global_cache_tags =
      tmpl::list<Tags::InitializeJ<Metavariables::evolve_ccm>>;
};

struct H5InitializeJMetavariables {
  // The initialization action picks the generator keyed on this, as the
  // production metavariables do.
  static constexpr bool evolve_ccm = false;
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using component_list = tmpl::list<
      mock_h5_worldtube_boundary_with_initialize_j<H5InitializeJMetavariables>>;
};

struct GhMetavariables {
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using component_list =
      tmpl::list<mock_worldtube_boundary<GhMetavariables,
                                         GhWorldtubeBoundary<GhMetavariables>>>;
};

struct AnalyticMetavariables {
  using cce_boundary_communication_tags =
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>;
  using component_list = tmpl::list<mock_worldtube_boundary<
      AnalyticMetavariables, AnalyticWorldtubeBoundary<AnalyticMetavariables>>>;
  using const_global_cache_tags = tmpl::list<
      Tags::CceEvolutionPrefix<::Tags::ConcreteTimeStepper<LtsTimeStepper>>>;
};

template <typename Generator>
void test_h5_initialization(const gsl::not_null<Generator*> gen) {
  using component =
      mock_worldtube_boundary<H5Metavariables,
                              H5WorldtubeBoundary<H5Metavariables>>;
  const size_t l_max = 8;
  const size_t end_time = 100.0;
  const size_t start_time = 0.0;
  const double extraction_radius = 100.0;
  ActionTesting::MockRuntimeSystem<H5Metavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<H5Metavariables>>{
          l_max, extraction_radius, end_time, start_time}};

  const size_t buffer_size = 8;
  const std::string filename = "InitializeWorldtubeBoundaryTest_CceR0100.h5";

  // create the test file, because on initialization the manager will need to
  // get basic data out of the file
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);
  TestHelpers::write_test_file(solution, filename, target_time,
                               extraction_radius, frequency, amplitude, l_max);

  ActionTesting::set_phase(make_not_null(&runner),
                           Parallel::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0,
      Tags::H5WorldtubeBoundaryDataManager::create_from_options(
          l_max, filename, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u),
          std::optional<double>{}));

  // this should run the initialization
  for (size_t i = 0; i < 3; ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);
  // check that the h5 data manager copied out of the databox has the correct
  // properties that we can examine without running the other actions
  const auto& data_manager =
      ActionTesting::get_databox_tag<component,
                                     Tags::H5WorldtubeBoundaryDataManager>(
          runner, 0);
  CHECK(data_manager.get_l_max() == l_max);
  const auto time_span = data_manager.get_time_span();
  CHECK(time_span.first == 0);
  CHECK(time_span.second == 0);

  // check that the Variables is in the expected state (here we just make sure
  // it has the right size - it shouldn't have been written to yet)
  const auto& variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename H5Metavariables::cce_boundary_communication_tags>>(runner,
                                                                      0);

  CHECK(get(get<Tags::BoundaryValue<Tags::BondiBeta>>(variables)).size() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}

// The `Du<Dr<J>>` boundary value is the initial data's alone, so the generator
// -- not `H5Interpolator` -- picks the interpolation order it is built with.
// This checks that the generator's order, and not merely some order, is the one
// the data manager in the box ends up using.
template <typename Generator>
void test_h5_du_dr_j_interpolator_injection(
    const gsl::not_null<Generator*> gen) {
  using component =
      mock_h5_worldtube_boundary_with_initialize_j<H5InitializeJMetavariables>;
  const size_t l_max = 8;
  const double end_time = 100.0;
  const double start_time = 0.0;
  const double extraction_radius = 100.0;
  const size_t buffer_size = 8;
  const std::string filename =
      "InitializeWorldtubeBoundaryDuDrJTest_CceR0100.h5";
  // Use six points for `H5Interpolator`, compared with two and four for the
  // requested orders. Orders 2 and 3 on four equally spaced points have
  // proportional barycentric weights and would give the same interpolant.
  const std::array<size_t, 2> du_dr_j_orders{{1, 2}};
  // The generator travels through the GlobalCache, which serializes it.
  register_derived_classes_with_charm<intrp::SpanInterpolator>();
  register_derived_classes_with_charm<InitializeJ::InitializeJ<false>>();

  // `operator()` is inherited from `std::uniform_real_distribution` and is
  // not const, so the distribution cannot be.
  // NOLINTNEXTLINE(misc-const-correctness)
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const gr::Solutions::KerrSchild solution{mass, spin, center};
  const double amplitude = 0.1 * value_dist(*gen);
  // Fix the phase and use 0.2 radians between samples (spaced by 0.1 in time)
  // so the interpolation errors remain distinguishable across random seeds.
  const double target_time = 5.0;
  const double frequency = 2.0;
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  TestHelpers::write_test_file(solution, filename, target_time,
                               extraction_radius, frequency, amplitude, l_max);

  const auto make_manager = [&]() {
    return Tags::H5WorldtubeBoundaryDataManager::create_from_options(
        l_max, filename, buffer_size,
        std::make_unique<intrp::BarycentricRationalSpanInterpolator>(4u, 5u),
        std::optional<double>{});
  };
  const auto boundary_data = [&target_time](const auto& manager) {
    Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
        variables{Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    // The lock is only held for the duration of the call; the manager does not
    // keep a reference to it.
    Parallel::NodeLock hdf5_lock{};
    manager.populate_hypersurface_boundary_data(
        make_not_null(&variables), target_time, make_not_null(&hdf5_lock));
    return get(get<Tags::BoundaryValue<Tags::Du<Tags::Dr<Tags::BondiJ>>>>(
                   variables))
        .data();
  };

  // What each candidate order gives when it is set on the data manager
  // directly, plus what `H5Interpolator` alone gives.
  std::unordered_map<size_t, ComplexDataVector> expected_by_order{};
  for (const size_t order : du_dr_j_orders) {
    auto manager = make_manager();
    manager->set_du_dr_j_interpolator(
        std::make_unique<intrp::BarycentricRationalSpanInterpolator>(order,
                                                                     order));
    expected_by_order.emplace(order, boundary_data(*manager));
  }
  const ComplexDataVector untouched_du_dr_j = boundary_data(*make_manager());

  for (const size_t du_dr_j_order : du_dr_j_orders) {
    CAPTURE(du_dr_j_order);
    ActionTesting::MockRuntimeSystem<H5InitializeJMetavariables> runner{
        tuples::tagged_tuple_from_typelist<
            Parallel::get_const_global_cache_tags<H5InitializeJMetavariables>>{
            std::make_unique<InitializeJ::CauchySecondOrder>(
                1.0e-11, 1500, true, 1.0e-1, 1.0e-14, 10, 1.0e-12,
                std::make_unique<intrp::BarycentricRationalSpanInterpolator>(
                    du_dr_j_order, du_dr_j_order)),
            l_max, extraction_radius, end_time, start_time}};
    ActionTesting::set_phase(make_not_null(&runner),
                             Parallel::Phase::Initialization);
    ActionTesting::emplace_component<component>(&runner, 0, make_manager());
    for (size_t i = 0; i < 3; ++i) {
      ActionTesting::next_action<component>(make_not_null(&runner), 0);
    }
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);

    const ComplexDataVector box_du_dr_j = boundary_data(
        ActionTesting::get_databox_tag<component,
                                       Tags::H5WorldtubeBoundaryDataManager>(
            runner, 0));

    // The manager the action left in the box gives exactly what the
    // generator's own order gives ...
    CHECK_ITERABLE_APPROX(box_du_dr_j, expected_by_order.at(du_dr_j_order));
    // ... and differs from the other candidate order and `H5Interpolator`
    // by more than numerical roundoff.
    const double scale = max(abs(box_du_dr_j));
    const auto differs_from = [&box_du_dr_j,
                               &scale](const ComplexDataVector& other) {
      return max(abs(box_du_dr_j - other)) > 1.0e-6 * scale;
    };
    for (const size_t other_order : du_dr_j_orders) {
      if (other_order != du_dr_j_order) {
        CAPTURE(other_order);
        CHECK(differs_from(expected_by_order.at(other_order)));
      }
    }
    CHECK(differs_from(untouched_du_dr_j));
  }

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}

void test_gh_initialization() {
  using component =
      mock_worldtube_boundary<GhMetavariables,
                              GhWorldtubeBoundary<GhMetavariables>>;
  const size_t l_max = 8;
  const double extraction_radius = 100.0;
  ActionTesting::MockRuntimeSystem<GhMetavariables> runner{
      tuples::tagged_tuple_from_typelist<
          Parallel::get_const_global_cache_tags<GhMetavariables>>{
          l_max, extraction_radius, std::numeric_limits<double>::infinity(),
          0.0}};

  runner.set_phase(Parallel::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0,
      InterfaceManagers::GhLocalTimeStepping{
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u)},
      InterfaceManagers::GhLockstep{});

  // this should run the initialization
  for (size_t i = 0; i < 3; ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }
  runner.set_phase(Parallel::Phase::Evolve);

  // check that the Variables is in the expected state (here we just make sure
  // it has the right size - it shouldn't have been written to yet)
  const auto& variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename GhMetavariables::cce_boundary_communication_tags>>(runner,
                                                                      0);
  CHECK(get(get<Tags::BoundaryValue<Tags::BondiBeta>>(variables)).size() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));
}

template <typename SolutionType>
void test_analytic_initialization() {
  using component =
      mock_worldtube_boundary<AnalyticMetavariables,
                              AnalyticWorldtubeBoundary<AnalyticMetavariables>>;
  const size_t l_max = 8;
  const double extraction_radius = 20.0;
  register_classes_with_charm<TimeSteppers::AdamsMoultonPc<false>>();
  ActionTesting::MockRuntimeSystem<AnalyticMetavariables> runner{
      {static_cast<std::unique_ptr<LtsTimeStepper>>(
           std::make_unique<::TimeSteppers::AdamsMoultonPc<false>>(3)),
       l_max, extraction_radius, 100.0, 0.0}};

  runner.set_phase(Parallel::Phase::Initialization);
  ActionTesting::emplace_component<component>(
      &runner, 0,
      AnalyticBoundaryDataManager{12_st, extraction_radius,
                                  std::make_unique<SolutionType>()});
  // this should run the initialization
  for (size_t i = 0; i < 3; ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), 0);
  }
  runner.set_phase(Parallel::Phase::Evolve);

  // check that the Variables is in the expected state (here we just make sure
  // it has the right size - it shouldn't have been written to yet)
  const auto& variables = ActionTesting::get_databox_tag<
      component,
      ::Tags::Variables<
          typename AnalyticMetavariables::cce_boundary_communication_tags>>(
      runner, 0);
  CHECK(get(get<Tags::BoundaryValue<Tags::BondiBeta>>(variables)).size() ==
        Spectral::Swsh::number_of_swsh_collocation_points(l_max));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.Actions.InitializeWorldtubeBoundary",
    "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_h5_initialization(make_not_null(&gen));
  test_h5_du_dr_j_interpolator_injection(make_not_null(&gen));
  test_gh_initialization();
  test_analytic_initialization<Solutions::RotatingSchwarzschild>();
  CHECK_THROWS_WITH(
      (test_analytic_initialization<Solutions::RobinsonTrautman>()),
      Catch::Matchers::ContainsSubstring(
          "Do not use RobinsonTrautman analytic solution with"));
}
}  // namespace Cce
