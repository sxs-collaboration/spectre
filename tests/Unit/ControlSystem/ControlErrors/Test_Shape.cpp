// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Component.hpp"
#include "ControlSystem/ControlErrors/Shape.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/Systems/Shape.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Frame {
struct Distorted;
}  // namespace Frame

namespace control_system {
namespace {
using FoTMap = std::unordered_map<
    std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
using Strahlkorper = Strahlkorper<Frame::Distorted>;

void test_shape_control_error() {
  constexpr size_t deriv_order = 2;
  using metavars =
      control_system::TestHelpers::MockMetavars<0, 0, 0, deriv_order>;
  using system = typename metavars::shape_system;
  using ControlError = system::control_error;
  using element_component = typename metavars::element_component;

  MAKE_GENERATOR(generator);
  domain::FunctionsOfTime::register_derived_with_charm();

  const tnsr::I<double, 3, Frame::Grid> origin{0.0};
  const double ah_radius = 1.5;
  const double initial_time = 0.0;
  Strahlkorper fake_ah{10, 10, make_array<double, 3>(origin)};
  auto& fake_ah_coefs = fake_ah.coefficients();

  // Setup initial shape map coefficients. In the map the coefficients are
  // stored as the negative of the actual spherical harmonic coefficients
  // because that's just how the map is defined. But since these are random
  // numbers it doesn't matter for initial data
  auto initial_shape_func = make_array<deriv_order + 1, DataVector>(
      DataVector{fake_ah_coefs.size(), 0.0});
  SpherepackIterator iter{fake_ah.l_max(), fake_ah.m_max()};
  std::uniform_real_distribution<double> coef_dist{-1.0, 1.0};
  for (size_t i = 0; i < initial_shape_func.size(); i++) {
    for (iter.reset(); iter; ++iter) {
      // Enforce l=0,l=1 components to be 0 always
      if (iter.l() == 0 or iter.l() == 1) {
        continue;
      }
      gsl::at(initial_shape_func, i)[iter()] = make_with_random_values<double>(
          make_not_null(&generator), coef_dist, 1);
    }
  }

  // Setup initial size map parameters
  auto initial_size_func =
      make_array<deriv_order + 1, DataVector>(DataVector{1, 0.0});
  // Excision sphere radius needs to be inside the AH
  const double excision_radius = 0.75 * ah_radius;
  // We only test for a constant size function of time. A test with a changing
  // size can be added in later
  initial_size_func[0][0] = ah_radius;

  // Setup control system stuff
  const std::string shape_name = system::name();
  const std::string size_name =
      ControlErrors::detail::size_name<::domain::ObjectLabel::A>();
  const std::string excision_sphere_A_name =
      ControlErrors::detail::excision_sphere_name<::domain::ObjectLabel::A>();
  const std::string excision_sphere_B_name =
      ControlErrors::detail::excision_sphere_name<::domain::ObjectLabel::B>();

  // Since the map for A/B are independent of each other, we only need to test
  // one of them
  FoTMap initial_functions_of_time{};
  FoTMap initial_measurement_timescales{};
  initial_functions_of_time[shape_name] = std::make_unique<
      domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>>(
      initial_time, initial_shape_func, 0.5);
  initial_functions_of_time[size_name] = std::make_unique<
      domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>>(
      initial_time, initial_size_func, std::numeric_limits<double>::infinity());

  // Fake domain
  Domain<3> fake_domain{
      {},
      {{excision_sphere_A_name,
        ExcisionSphere<3>{excision_radius,
                          origin,
                          {{0, Direction<3>::lower_zeta()},
                           {1, Direction<3>::lower_zeta()},
                           {2, Direction<3>::lower_zeta()},
                           {3, Direction<3>::lower_zeta()},
                           {4, Direction<3>::lower_zeta()},
                           {5, Direction<3>::lower_zeta()}}}},
       {excision_sphere_B_name,
        ExcisionSphere<3>{excision_radius,
                          origin,
                          {{0, Direction<3>::lower_zeta()},
                           {1, Direction<3>::lower_zeta()},
                           {2, Direction<3>::lower_zeta()},
                           {3, Direction<3>::lower_zeta()},
                           {4, Direction<3>::lower_zeta()},
                           {5, Direction<3>::lower_zeta()}}}}}};

  auto grid_center_A =
      fake_domain.excision_spheres().at("ExcisionSphereA").center();
  auto grid_center_B =
      fake_domain.excision_spheres().at("ExcisionSphereA").center();

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  // Excision centers aren't used so their values can be anything
  MockRuntimeSystem runner{
      {"DummyFilename", std::move(fake_domain), 4, false, ::Verbosity::Silent,
       std::move(grid_center_A), std::move(grid_center_B)},
      {std::move(initial_functions_of_time),
       std::move(initial_measurement_timescales)}};
  ActionTesting::emplace_array_component<element_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);

  auto& cache = ActionTesting::cache<element_component>(runner, 0);
  const auto& functions_of_time =
      Parallel::get<domain::Tags::FunctionsOfTime>(cache);

  const double check_time = 0.1;

  const auto measurement_coefs = make_with_random_values<DataVector>(
      make_not_null(&generator), coef_dist, fake_ah_coefs);
  fake_ah_coefs = measurement_coefs;

  using QueueTuple =
      tuples::TaggedTuple<control_system::QueueTags::Horizon<Frame::Distorted>>;
  QueueTuple fake_measurement_tuple{fake_ah};

  const DataVector control_error =
      ControlError{}(cache, check_time, shape_name, fake_measurement_tuple);

  const auto lambda_00_coef =
      functions_of_time.at(size_name)->func(check_time)[0][0];
  const double Y00 = sqrt(0.25 / M_PI);
  const auto lambda_lm_coefs =
      functions_of_time.at(shape_name)->func(check_time)[0];

  DataVector expected_control_error =
      -(excision_radius / Y00 - lambda_00_coef) /
          (sqrt(0.5 * M_PI) * measurement_coefs[iter.set(0, 0)()]) *
          measurement_coefs -
      lambda_lm_coefs;
  // We don't control l=0 or l=1 modes
  for (iter.reset(); iter; ++iter) {
    if (iter.l() == 0 or iter.l() == 1) {
      expected_control_error[iter()] = 0.0;
    }
  }

  CHECK_ITERABLE_APPROX(control_error, expected_control_error);
}

SPECTRE_TEST_CASE("Unit.ControlSystem.ControlErrors.Shape",
                  "[ControlSystem][Unit]") {
  test_shape_control_error();
}
}  // namespace
}  // namespace control_system
