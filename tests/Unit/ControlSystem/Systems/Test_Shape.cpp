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
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/ControlSystem/SystemHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/IO/Observers/MockWriteReductionDataRow.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeString.hpp"
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
template <typename Metavars>
using SystemHelper = control_system::TestHelpers::SystemHelper<Metavars>;

template <typename Generator, typename Metavars, size_t DerivOrder>
void test_shape_control(
    const gsl::not_null<Generator*> generator,
    const gsl::not_null<SystemHelper<Metavars>*> system_helper,
    const double initial_time, const double final_time, const size_t l_max,
    const domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder>&
        ah_coefs_function_of_time,
    const double func_eps, const double deriv_eps) {
  using system = typename Metavars::shape_system;
  using shape_component = typename Metavars::shape_component;
  using element_component = typename Metavars::element_component;

  auto& domain = system_helper->domain();
  auto& initial_functions_of_time = system_helper->initial_functions_of_time();
  auto& initial_measurement_timescales =
      system_helper->initial_measurement_timescales();

  auto grid_center_A = domain.excision_spheres().at("ExcisionSphereA").center();
  auto grid_center_B = domain.excision_spheres().at("ExcisionSphereB").center();

  const auto& init_shape_tuple = system_helper->template init_tuple<system>();

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavars>;
  // Excision centers aren't used so their values can be anything
  MockRuntimeSystem runner{
      {"DummyFileName", std::move(domain), 4, false, ::Verbosity::Silent,
       std::move(grid_center_A), std::move(grid_center_B)},
      {std::move(initial_functions_of_time),
       std::move(initial_measurement_timescales)}};
  ActionTesting::emplace_singleton_component_and_initialize<shape_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, init_shape_tuple);
  ActionTesting::emplace_array_component<element_component>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, 0);

  runner.set_phase(Parallel::Phase::Testing);

  const auto& cache = ActionTesting::cache<element_component>(runner, 0);
  const auto& cache_domain = get<::domain::Tags::Domain<3>>(cache);
  const auto& excision_sphere = cache_domain.excision_spheres().at(
      control_system::ControlErrors::detail::excision_sphere_name<
          ::domain::ObjectLabel::A>());

  SpherepackIterator iter{l_max, l_max};
  const double ah_radius =
      ah_coefs_function_of_time.func(initial_time)[0][iter.set(0, 0)()];
  const tnsr::I<double, 3, Frame::Grid>& grid_center = excision_sphere.center();
  Strahlkorper horizon_a{l_max, l_max, ah_radius,
                         make_array<double, 3>(grid_center)};
  // B just needs to exist. Doesn't have to be valid
  const Strahlkorper horizon_b{};

  const auto horizon_measurement =
      [&horizon_a, &horizon_b, &ah_coefs_function_of_time](const double time) {
        auto& mutable_horizon_coefficients = horizon_a.coefficients();
        mutable_horizon_coefficients =
            ah_coefs_function_of_time.func_and_2_derivs(time)[0];

        return std::pair<Strahlkorper, Strahlkorper>{horizon_a, horizon_b};
      };

  // Run the test, specifying that we are only using 1 horizon
  system_helper->run_control_system_test(runner, final_time, generator,
                                         horizon_measurement);

  const auto& functions_of_time =
      Parallel::get<domain::Tags::FunctionsOfTime>(cache);
  const double excision_radius = excision_sphere.radius();
  const std::string size_name =
      control_system::ControlErrors::detail::size_name<
          ::domain::ObjectLabel::A>();
  const std::string shape_name = system_helper->template name<system>();

  const auto lambda_00_coef =
      functions_of_time.at(size_name)->func(final_time)[0][0];
  const double Y00 = sqrt(0.25 / M_PI);

  auto ah_coefs_and_derivs =
      ah_coefs_function_of_time.func_and_2_derivs(final_time);

  // Our expected coefs are just the (minus) coefs of the AH (except for l=0,l=1
  // which should be zero) scaled by the relative size factor defined in the
  // control error
  auto expected_shape_coefs =
      -1.0 * (excision_radius / Y00 - lambda_00_coef) /
      (sqrt(0.5 * M_PI) * ah_coefs_and_derivs[0][iter.set(0, 0)()]) *
      ah_coefs_and_derivs;
  // Manually set 0,0 component to 0
  expected_shape_coefs[0][iter.set(0, 0)()] = 0.0;

  const auto& shape_func =
      dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder>&>(
          *functions_of_time.at(shape_name));

  const auto shape_coefs = shape_func.func_and_2_derivs(final_time);

  Approx custom_approx = Approx::custom().epsilon(1.0).scale(1.0);
  for (size_t i = 0; i < shape_coefs.size(); i++) {
    if (i == 0) {
      custom_approx = Approx::custom().epsilon(func_eps).scale(1.0);
    } else {
      custom_approx = Approx::custom().epsilon(deriv_eps).scale(1.0);
    }
    INFO("i = " + get_output(i));
    CHECK_ITERABLE_CUSTOM_APPROX(gsl::at(shape_coefs, i),
                                 gsl::at(expected_shape_coefs, i),
                                 custom_approx);
  }
}

template <size_t DerivOrder, typename Generator>
void test_suite(const gsl::not_null<Generator*> generator, const size_t l_max,
                const double looser_eps, const double stricter_eps) {
  // First 3 zeros are for translation, rotation, and expansion
  using metavars =
      control_system::TestHelpers::MockMetavars<0, 0, 0, DerivOrder>;
  using system = typename metavars::shape_system;

  // Responsible for running all control system checks
  SystemHelper<metavars> system_helper{};
  SpherepackIterator iter{l_max, l_max};
  const size_t num_ah_coeffs = iter.spherepack_array_size();
  std::uniform_real_distribution<double> coef_dist{-0.1, 0.1};
  // Hard code the radius to something other than 1 because shape doesn't depend
  // on the size of the strahlkorper
  const double ah_radius = 1.63;
  const double initial_time = 0.0;
  const double final_time = 100.0;

  // Setup initial shape map coefficients. In the map the coefficients are
  // stored as the negative of the actual spherical harmonic coefficients
  // because that's just how the map is defined. But since these are random
  // numbers it doesn't matter for initial data. Also set initial size map
  // coefficient. Here we only test non-changing size. A test with a changing
  // size parameter can be added in later if needed.
  const auto initialize_shape_functions_of_time =
      [&iter, &generator, &coef_dist, &ah_radius](
          const gsl::not_null<std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
              functions_of_time,
          const double local_initial_time,
          const std::unordered_map<std::string, double>&
              initial_expiration_times) {
        auto initial_shape_func = make_array<DerivOrder + 1, DataVector>(
            DataVector{iter.spherepack_array_size(), 0.0});

        for (size_t i = 0; i < initial_shape_func.size(); i++) {
          for (iter.reset(); iter; ++iter) {
            // Enforce l=0,l=1 components to be 0 always
            if (iter.l() == 0 or iter.l() == 1) {
              continue;
            }
            gsl::at(initial_shape_func, i)[iter()] =
                make_with_random_values<double>(generator, coef_dist, 1.0);
          }
        }

        const std::string shape_name = system::name();
        (*functions_of_time)[shape_name] = std::make_unique<
            domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder>>(
            local_initial_time, initial_shape_func,
            initial_expiration_times.at(shape_name));

        auto initial_size_func =
            make_array<DerivOrder + 1, DataVector>(DataVector{1, 0.0});
        initial_size_func[0][0] = ah_radius;
        const std::string size_name =
            ControlErrors::detail::size_name<::domain::ObjectLabel::A>();
        (*functions_of_time)[size_name] = std::make_unique<
            domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder>>(
            local_initial_time, initial_size_func,
            std::numeric_limits<double>::infinity());

        // Excision sphere radius just needs to be inside the horizon. In an
        // actual run we'd want the excision surface closer to the horizon
        return 0.75 * ah_radius;
      };

  const std::string num_ah_coeffs_str = MakeString{} << num_ah_coeffs;

  const std::string input_options =
      "Evolution:\n"
      "  InitialTime: 0.0\n"
      "DomainCreator:\n"
      "  FakeCreator:\n"
      "    NumberOfExcisions: 1\n"
      "    NumberOfComponents:\n"
      "      ShapeA: " +
      num_ah_coeffs_str +
      "\n"
      "ControlSystems:\n"
      "  WriteDataToDisk: false\n"
      "  MeasurementsPerUpdate: 4\n"
      "  ShapeA:\n"
      "    IsActive: true\n"
      "    Averager:\n"
      "      AverageTimescaleFraction: 0.25\n"
      "      Average0thDeriv: true\n"
      "    Controller:\n"
      "      UpdateFraction: 0.3\n"
      "    TimescaleTuner:\n"
      "      InitialTimescales: 0.3\n"
      "      MinTimescale: 0.1\n"
      "      MaxTimescale: 10.\n"
      "      DecreaseThreshold: 4e-3\n"
      "      IncreaseThreshold: 1e-3\n"
      "      IncreaseFactor: 1.01\n"
      "      DecreaseFactor: 0.98\n"
      "    ControlError:\n";

  // Separation doesn't matter because we are only using AhA
  system_helper.setup_control_system_test(initial_time, 15.0, input_options,
                                          initialize_shape_functions_of_time);

  const DataVector zero_dv{num_ah_coeffs, 0.0};
  const auto zero_array = make_array<DerivOrder + 1, DataVector>(zero_dv);

  // Allocate now and reuse as we go
  auto initial_ah_coefs = zero_array;
  domain::FunctionsOfTime::PiecewisePolynomial<DerivOrder> expected_ah_coefs{};

  const std::string deriv_order_string =
      "DerivOrder=" + get_output(DerivOrder) + ": ";
  {
    INFO(deriv_order_string + "Stationary spherical AH");
    // Set all coefs back to zero first
    initial_ah_coefs = zero_array;
    initial_ah_coefs[0][iter.set(0, 0)()] = ah_radius;
    expected_ah_coefs = {initial_time, initial_ah_coefs,
                         std::numeric_limits<double>::infinity()};
    test_shape_control(generator, make_not_null(&system_helper), initial_time,
                       final_time, l_max, expected_ah_coefs, stricter_eps,
                       stricter_eps);
  }
  system_helper.reset();
  {
    INFO(deriv_order_string + "Stationary non-spherical AH");
    // Set all coefs back to zero first
    initial_ah_coefs = zero_array;
    for (iter.reset(); iter; ++iter) {
      // Enforce l=0,l=1 components to be 0 always
      if (iter.l() == 0 or iter.l() == 1) {
        continue;
      }
      initial_ah_coefs[0][iter()] =
          make_with_random_values<double>(generator, coef_dist, 1);
    }
    // Ensure that radius of AH is positive and constant
    initial_ah_coefs[0][iter.set(0, 0)()] = ah_radius;
    initial_ah_coefs[1] = zero_dv;
    initial_ah_coefs[2] = zero_dv;
    expected_ah_coefs = {initial_time, initial_ah_coefs,
                         std::numeric_limits<double>::infinity()};
    test_shape_control(generator, make_not_null(&system_helper), initial_time,
                       final_time, l_max, expected_ah_coefs, stricter_eps,
                       stricter_eps);
  }
  system_helper.reset();
  {
    INFO(deriv_order_string + "AH coefficients linearly increasing/decreasing");
    // Set all coefs back to zero first
    initial_ah_coefs = zero_array;
    for (iter.reset(); iter; ++iter) {
      // Enforce l=0,l=1 components to be 0 always
      if (iter.l() == 0 or iter.l() == 1) {
        continue;
      }
      initial_ah_coefs[0][iter()] =
          make_with_random_values<double>(generator, coef_dist, 1);
      initial_ah_coefs[1][iter()] =
          make_with_random_values<double>(generator, coef_dist, 1);
    }
    // Ensure that radius of AH is positive and constant
    initial_ah_coefs[0][iter.set(0, 0)()] = ah_radius;
    initial_ah_coefs[1][iter.set(0, 0)()] = 0.0;
    initial_ah_coefs[2] = zero_dv;
    expected_ah_coefs = {initial_time, initial_ah_coefs,
                         std::numeric_limits<double>::infinity()};
    test_shape_control(generator, make_not_null(&system_helper), initial_time,
                       final_time, l_max, expected_ah_coefs, stricter_eps,
                       stricter_eps);
  }
  system_helper.reset();
  {
    INFO(deriv_order_string +
         "AH coefficients quadratically increasing/decreasing");
    // Set all coefs back to zero first
    initial_ah_coefs = zero_array;
    for (iter.reset(); iter; ++iter) {
      // Enforce l=0,l=1 components to be 0 always
      if (iter.l() == 0 or iter.l() == 1) {
        continue;
      }
      for (size_t i = 0; i < 3; i++) {
        gsl::at(initial_ah_coefs, i)[iter()] =
            make_with_random_values<double>(generator, coef_dist, 1);
      }
    }
    // Ensure that radius of AH is positive and constant
    initial_ah_coefs[0][iter.set(0, 0)()] = ah_radius;
    initial_ah_coefs[1][iter.set(0, 0)()] = 0.0;
    initial_ah_coefs[2][iter.set(0, 0)()] = 0.0;
    expected_ah_coefs = {initial_time, initial_ah_coefs,
                         std::numeric_limits<double>::infinity()};
    test_shape_control(generator, make_not_null(&system_helper), initial_time,
                       final_time, l_max, expected_ah_coefs, looser_eps,
                       looser_eps);
  }
  system_helper.reset();
  {
    INFO(deriv_order_string + "AH coefficients oscillating sinusoidally");
    // Set all coefs back to zero first
    initial_ah_coefs = zero_array;
    // All coefficients (and derivatives) are of the form
    // coef = sin(freq * time + offset)
    // dtcoef = freq * cos(freq * time + offset)
    // d2tcoef = -freq^2 * sin(freq * time + offset)
    // d3tcoef = -freq^3 * cos(freq * time + offset)

    // Only allow at most one full oscillation by the time we reach the end.
    // That way the coefficients aren't oscillating too fast
    std::uniform_real_distribution<double> freq_dist{0.0,
                                                     2.0 * M_PI / final_time};
    const double amplitude = 0.1;
    DataVector freqs{zero_dv};
    DataVector offsets{zero_dv};
    for (iter.reset(); iter; ++iter) {
      // Enforce l=0,l=1 components to be 0 always
      if (iter.l() == 0 or iter.l() == 1) {
        continue;
      }
      const double offset =
          make_with_random_values<double>(generator, coef_dist, 1);
      const double freq =
          make_with_random_values<double>(generator, freq_dist, 1);
      freqs[iter()] = freq;
      offsets[iter()] = offset;
      initial_ah_coefs[0][iter()] = sin(freq * initial_time + offset);
      initial_ah_coefs[1][iter()] = freq * cos(freq * initial_time + offset);
      initial_ah_coefs[2][iter()] =
          -square(freq) * sin(freq * initial_time + offset);
      if (DerivOrder > 2) {
        initial_ah_coefs[3][iter()] =
            -cube(freq) * cos(freq * initial_time + offset);
      }
    }
    // Ensure that radius of AH is positive and constant
    initial_ah_coefs = amplitude * initial_ah_coefs;
    initial_ah_coefs[0][iter.set(0, 0)()] = ah_radius;
    initial_ah_coefs[1][iter.set(0, 0)()] = 0.0;
    initial_ah_coefs[2][iter.set(0, 0)()] = 0.0;
    if (DerivOrder > 2) {
      initial_ah_coefs[3][iter.set(0, 0)()] = 0.0;
    }

    // Initialize the expected function of time
    const double dt = 0.1;
    double time = initial_time + dt;
    expected_ah_coefs = {initial_time, initial_ah_coefs, time};

    // Update it's derivative often so the function is smooth
    DataVector updated_deriv = zero_dv;

    const auto update_deriv = [&updated_deriv, &iter, &time, &amplitude, &freqs,
                               &offsets]() {
      for (iter.reset(); iter; ++iter) {
        if (iter.l() == 0 or iter.l() == 1) {
          continue;
        }
        const double freq = freqs[iter()];
        const double offset = offsets[iter()];
        if (DerivOrder == 2) {
          updated_deriv[iter()] =
              -amplitude * square(freq) * sin(freq * time + offset);
        } else {
          updated_deriv[iter()] =
              -amplitude * cube(freq) * cos(freq * time + offset);
        }
      }
    };

    while (time < final_time) {
      update_deriv();
      expected_ah_coefs.update(time, updated_deriv, time + dt);
      time += dt;
    }

    // Use looser_eps for both because the function is not perfectly represented
    // by a polynomial
    test_shape_control(generator, make_not_null(&system_helper), initial_time,
                       final_time, l_max, expected_ah_coefs, looser_eps,
                       looser_eps);
  }
}

void test_names() {
  using shape = control_system::Systems::Shape<::domain::ObjectLabel::A, 2,
                                               measurements::BothHorizons>;

  CHECK(pretty_type::name<shape>() == "ShapeA");

  const size_t l_max = 3;
  SpherepackIterator iter(l_max, l_max);
  const size_t size = iter.spherepack_array_size();

  // Check known valid indices
  for (iter.reset(); iter; ++iter) {
    const auto component_name = shape::component_name(iter(), size);
    CHECK(component_name.has_value());
    const int m = iter() < size / 2 ? static_cast<int>(iter.m())
                                    : -static_cast<int>(iter.m());
    const std::string check_name =
        "l"s + get_output(iter.l()) + "m"s + get_output(m);
    CHECK(*component_name == check_name);
  }

  // We hard code the names for this specific ell so we know they are right. The
  // (--) are just place holders for spherepack components that don't correspond
  // to an l,m. They won't be checked (if everything is working properly)
  const std::vector<std::string> expected_names{
      "l0m0", "(--)",  "(--)",  "(--)", "l1m0", "l1m1",  "(--)",  "(--)",
      "l2m0", "l2m1",  "l2m2",  "(--)", "l3m0", "l3m1",  "l3m2",  "l3m3",
      "(--)", "(--)",  "(--)",  "(--)", "(--)", "l1m-1", "(--)",  "(--)",
      "(--)", "l2m-1", "l2m-2", "(--)", "(--)", "l3m-1", "l3m-2", "l3m-3"};

  CHECK(size == expected_names.size());

  // Check all indices
  iter.reset();
  for (size_t i = 0; i < size; i++) {
    const auto compact_index = iter.compact_index(i);
    const auto component_name = shape::component_name(i, size);
    CHECK(compact_index.has_value() == component_name.has_value());
    if (component_name.has_value()) {
      CHECK(*compact_index == iter.current_compact_index());
      const std::string& check_name = expected_names[i];
      CHECK(*component_name == check_name);
      ++iter;
    }
  }
}

// [[TimeOut, 30]]
SPECTRE_TEST_CASE("Unit.ControlSystem.Systems.Shape", "[ControlSystem][Unit]") {
  MAKE_GENERATOR(generator);
  domain::FunctionsOfTime::register_derived_with_charm();

  test_names();

  // For some of the AHs (quadratic and sinusoid), the control system isn't able
  // to perfectly match the expected map parameters, but it is able to get the
  // control error to be a small constant offset (rather than 0). This larger
  // epsilon accounts for the difference in the map parameters based off the
  // constant offset in the control error.
  const double custom_approx_looser_eps = 5.0e-3;
  // For everything else, we can use a stricter epsilon
  const double custom_approx_stricter_eps = 5.0e-7;

  std::vector<size_t> ells_to_test{2, 5};
  for (const auto& ell : ells_to_test) {
    test_suite<2>(make_not_null(&generator), ell, custom_approx_looser_eps,
                  custom_approx_stricter_eps);
    test_suite<3>(make_not_null(&generator), ell, custom_approx_looser_eps,
                  custom_approx_stricter_eps);
  }
}
}  // namespace
}  // namespace control_system
