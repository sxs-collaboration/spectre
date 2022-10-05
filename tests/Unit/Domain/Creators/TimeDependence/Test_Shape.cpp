// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/Creators/TimeDependence/Shape.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/Creators/TimeDependence/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependence {

namespace {
using ShapeMap = domain::CoordinateMaps::TimeDependent::Shape;

using ConcreteMap =
    domain::CoordinateMap<Frame::Grid, Frame::Inertial, ShapeMap>;

using Transition =
    domain::CoordinateMaps::ShapeMapTransitionFunctions::SphereTransition;

ConcreteMap create_coord_map(
    const std::string& f_of_t_name, const size_t l_max,
    const std::array<double, 3>& center,
    std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                        ShapeMapTransitionFunction>
        transition_func) {
  return ConcreteMap{{ShapeMap{center, l_max, l_max,
                               transition_func->get_clone(), f_of_t_name}}};
}

template <typename Frame>
std::array<double, 3> r_theta_phi(const tnsr::I<double, 3, Frame>& input) {
  const double input_rho = sqrt(square(input.get(0)) + square(input.get(1)));
  return {{magnitude(input).get(), atan2(input_rho, input.get(2)),
           atan2(input.get(1), input.get(0))}};
}

// Takes an input point in Frame::Grid and an output point in Frame::Inertial
// produced by a Shape CoordinateMap and compares it to an expected output
// point corresponding to what one would obtain if an analytic mapping were
// used instead of an expansion over spherical harmonics, as is used by the
// Shape map. For an `l_max` of 16, this test can be expected to pass for any
// random point mapped for a Shape map with dim'less spin magnitude < 0.65.
void test_r_theta_phi(const tnsr::I<double, 3, Frame::Grid>& input,
                      const tnsr::I<double, 3, Frame::Inertial>& output,
                      const double inner_radius, const double outer_radius,
                      const double mass, const std::array<double, 3>& spin,
                      const std::array<double, 3>& center) {
  auto input_centered =
      make_with_value<tnsr::I<double, 3, Frame::Grid>>(input, 0.0);
  auto output_centered =
      make_with_value<tnsr::I<double, 3, Frame::Inertial>>(output, 0.0);
  for (size_t i = 0; i < 3; i++) {
    input_centered.get(i) = input.get(i) - gsl::at(center, i);
    output_centered.get(i) = output.get(i) - gsl::at(center, i);
  }
  const auto input_centered_spherical = r_theta_phi(input_centered);
  const auto output_centered_spherical = r_theta_phi(output_centered);
  const std::array<double, 2> input_theta_phi = {
      {input_centered_spherical[1], input_centered_spherical[2]}};
  CHECK(input_theta_phi[0] == approx(output_centered_spherical[1]));
  CHECK(input_theta_phi[1] == approx(output_centered_spherical[2]));
  const double kerr_schild_radius =
      gr::Solutions::kerr_schild_radius_from_boyer_lindquist(
          inner_radius, input_theta_phi, mass, spin)
          .get();
  const double transition_factor =
      std::make_unique<Transition>(Transition{inner_radius, outer_radius})
          ->
          operator()({{magnitude(input_centered).get(), 0.0, 0.0}});
  const double expected_output_centered_spherical =
      input_centered_spherical[0] *
      (1.0 +
       transition_factor * (kerr_schild_radius - inner_radius) / inner_radius);
  CHECK(output_centered_spherical[0] ==
        approx(expected_output_centered_spherical));
}

void test(const std::unique_ptr<TimeDependence<3>>& time_dep_unique_ptr,
          const double initial_time, const std::string& f_of_t_name,
          const size_t l_max,
          std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                              ShapeMapTransitionFunction>
              transition_func,
          const double inner_radius, const double outer_radius,
          const double mass, const std::array<double, 3>& spin,
          const std::array<double, 3>& center) {
  MAKE_GENERATOR(gen);
  CAPTURE(initial_time);
  CAPTURE(f_of_t_name);
  CAPTURE(l_max);
  CAPTURE(spin);
  CAPTURE(center);

  CHECK_FALSE(time_dep_unique_ptr->is_none());

  // We downcast to the expected derived class to make sure that factory
  // creation worked correctly. In order to maximize code reuse this check is
  // done here as opposed to separately elsewhere.
  const auto* const time_dep =
      dynamic_cast<const Shape*>(time_dep_unique_ptr.get());
  REQUIRE(time_dep != nullptr);

  // Test coordinate maps
  UniformCustomDistribution<size_t> dist_size_t{1, 10};
  const size_t num_blocks = dist_size_t(gen);
  CAPTURE(num_blocks);
  const auto expected_block_map = create_coord_map(
      f_of_t_name, l_max, center, transition_func->get_clone());
  const auto block_maps =
      time_dep_unique_ptr->block_maps_grid_to_inertial(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps) {
    const auto* const block_map =
        dynamic_cast<const ConcreteMap*>(block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_block_map);
  }

  // Check that maps involving distorted frame are empty.
  CHECK(time_dep_unique_ptr->block_maps_grid_to_distorted(1)[0].get() ==
        nullptr);
  CHECK(time_dep_unique_ptr->block_maps_distorted_to_inertial(1)[0].get() ==
        nullptr);

  // Test functions of time without expiration times
  {
    const auto functions_of_time = time_dep_unique_ptr->functions_of_time();
    REQUIRE(functions_of_time.size() == 1);
    CHECK(functions_of_time.count(f_of_t_name) == 1);
    CHECK(functions_of_time.at(f_of_t_name)->time_bounds()[1] ==
          std::numeric_limits<double>::infinity());
  }
  // Test functions of time with expiration times
  {
    const double init_expr_time = 5.0;
    std::unordered_map<std::string, double> init_expr_times{};
    init_expr_times[f_of_t_name] = init_expr_time;
    const auto functions_of_time =
        time_dep_unique_ptr->functions_of_time(init_expr_times);
    REQUIRE(functions_of_time.size() == 1);
    CHECK(functions_of_time.count(f_of_t_name) == 1);
    CHECK(functions_of_time.at(f_of_t_name)->time_bounds()[1] ==
          init_expr_time);
  }

  const auto functions_of_time = time_dep_unique_ptr->functions_of_time();

  // For a random point at a random time check that the values agree. This is
  // to check that the internals were assigned the correct function of times.
  // The points are are drawn from the positive x/y/z quadrant such that their
  // radii lie between inner_radius and outer_radius.
  TIME_DEPENDENCE_GENERATE_COORDS(make_not_null(&gen), 3,
                                  inner_radius / sqrt(3.0),
                                  outer_radius / sqrt(3.0));

  for (const auto& block_map : block_maps) {
    // We've checked equivalence above
    // (CHECK(*block_map == expected_block_map);), but have sometimes been
    // burned by incorrect operator== implementations so we check that the
    // mappings behave as expected.
    const double check_time = initial_time + dist(gen) + 1.2;
    CHECK_ITERABLE_APPROX(
        expected_block_map(grid_coords_dv, check_time, functions_of_time),
        (*block_map)(grid_coords_dv, check_time, functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_map(grid_coords_double, check_time, functions_of_time),
        (*block_map)(grid_coords_double, check_time, functions_of_time));

    CHECK_ITERABLE_APPROX(
        *expected_block_map.inverse(inertial_coords_double, check_time,
                                    functions_of_time),
        *block_map->inverse(inertial_coords_double, check_time,
                            functions_of_time));

    CHECK_ITERABLE_APPROX(
        expected_block_map.inv_jacobian(grid_coords_dv, check_time,
                                        functions_of_time),
        block_map->inv_jacobian(grid_coords_dv, check_time, functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_map.inv_jacobian(grid_coords_double, check_time,
                                        functions_of_time),
        block_map->inv_jacobian(grid_coords_double, check_time,
                                functions_of_time));

    CHECK_ITERABLE_APPROX(
        expected_block_map.jacobian(grid_coords_dv, check_time,
                                    functions_of_time),
        block_map->jacobian(grid_coords_dv, check_time, functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_map.jacobian(grid_coords_double, check_time,
                                    functions_of_time),
        block_map->jacobian(grid_coords_double, check_time, functions_of_time));
    // Initialize a non spherical shape and make sure the values match
    // kerr map takes mass and spin, gives radius at different theta phis
    // get modal coefficients from SPHEREPACK (ylms corresponding to r(theta,
    // phi) keep in mind how this interacts with r==0, consider an assert.
    const auto output_point =
        (*block_map)(grid_coords_double, check_time, functions_of_time);
    test_r_theta_phi(grid_coords_double, output_point, inner_radius,
                     outer_radius, mass, spin, center);
  }
}

void test_equivalence() {
  const double mass = 1.0;
  const std::array<double, 3> spin1 = {{0.0, 0.5, 0.1}};
  const std::array<double, 3> spin2 = {{0.3, 0.2, 0.0}};
  const std::array<double, 3> center1 = {{-0.2, 1.3, 2.4}};
  const std::array<double, 3> center2 = {{0.2, -1.3, 2.4}};
  const double inner_radius = 1.0;
  const double outer_radius1 = 10.0;
  const double outer_radius2 = 20.0;

  Shape sc0{1.0, 4, mass, spin1, center1, inner_radius, outer_radius1};
  Shape sc1{1.0, 4, mass, spin1, center1, inner_radius, outer_radius1};
  Shape sc2{1.0, 4, mass, spin2, center1, inner_radius, outer_radius1};
  Shape sc3{1.0, 4, mass, spin2, center1, inner_radius, outer_radius1};
  Shape sc4{1.0, 4, mass, spin1, center2, inner_radius, outer_radius2};
  Shape sc5{1.0, 4, mass, spin1, center2, inner_radius, outer_radius2};
  Shape sc6{1.0, 4, mass, spin2, center2, inner_radius, outer_radius2};
  Shape sc7{1.0, 4, mass, spin2, center2, inner_radius, outer_radius2};

  CHECK(sc0 == sc0);
  CHECK_FALSE(sc0 != sc0);
  CHECK(sc0 == sc1);
  CHECK_FALSE(sc0 != sc1);
  CHECK(sc0 != sc2);
  CHECK_FALSE(sc0 == sc2);
  CHECK(sc0 != sc3);
  CHECK_FALSE(sc0 == sc3);
  CHECK(sc0 != sc4);
  CHECK_FALSE(sc0 == sc4);
  CHECK(sc0 != sc5);
  CHECK_FALSE(sc0 == sc5);
  CHECK(sc0 != sc6);
  CHECK_FALSE(sc0 == sc6);
  CHECK(sc0 != sc7);
  CHECK_FALSE(sc0 == sc7);
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependence.Shape",
                  "[Domain][Unit]") {
  constexpr double initial_time{1.3};
  // l_max of 16 needed to pass tests using `approx` as tests
  // compare ylm output with analytic solution.
  constexpr size_t l_max{16};
  constexpr double mass{1.0};
  const std::array<double, 3> spin{{0.1, 0.4, -0.5}};
  const std::array<double, 3> center{{-0.02, 0.013, 0.024}};
  const double inner_radius = 2.0;
  const double outer_radius = 100;
  const Transition sphere_transition{inner_radius, outer_radius};
  // This name must match the hard coded one in Shape
  const std::string f_of_t_name = "Shape";

  const std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dep = std::make_unique<Shape>(initial_time, l_max, mass, spin,
                                         center, inner_radius, outer_radius);
  test(time_dep, initial_time, f_of_t_name, l_max,
       std::make_unique<Transition>(sphere_transition), inner_radius,
       outer_radius, mass, spin, center);
  test(time_dep->get_clone(), initial_time, f_of_t_name, l_max,
       std::make_unique<Transition>(sphere_transition), inner_radius,
       outer_radius, mass, spin, center);

  test(TestHelpers::test_creation<std::unique_ptr<TimeDependence<3>>>(
           "Shape:\n"
           "  InitialTime: 1.3\n"
           "  LMax: 16\n"
           "  Mass: 1.0\n"
           "  Spin: [0.1, 0.4, -0.5]\n"
           "  Center: [-0.02, 0.013, 0.024]\n"
           "  InnerRadius: 2.0\n"
           "  OuterRadius: 100.0\n"),
       initial_time, f_of_t_name, l_max,
       std::make_unique<Transition>(sphere_transition), inner_radius,
       outer_radius, mass, spin, center);
  test_equivalence();

  CHECK_THROWS_WITH(
      TestHelpers::test_creation<std::unique_ptr<TimeDependence<3>>>(
          "Shape:\n"
          "  InitialTime: 1.3\n"
          "  LMax: 4\n"
          "  Mass: 1.0\n"
          "  Spin: [0.0, 1.0, 0.1]\n"
          "  Center: [-0.01, 0.02, 0.01]\n"
          "  InnerRadius: 1.0\n"
          "  OuterRadius: 10.0\n"),
      Catch::Contains("Tried to create a Shape TimeDependence, but the "
                      "magnitude of the spin"));

  CHECK_THROWS_WITH(
      TestHelpers::test_creation<std::unique_ptr<TimeDependence<3>>>(
          "Shape:\n"
          "  InitialTime: 1.3\n"
          "  LMax: 4\n"
          "  Mass: 1.0\n"
          "  Spin: [0.0, 1.0, 0.1]\n"
          "  Center: [-0.01, 0.02, 0.01]\n"
          "  InnerRadius: 10.0\n"
          "  OuterRadius: 1.0\n"),
      Catch::Contains(
          "The maximum radius must be greater than the minimum radius"));

  CHECK_THROWS_WITH(
      TestHelpers::test_creation<std::unique_ptr<TimeDependence<3>>>(
          "Shape:\n"
          "  InitialTime: 1.3\n"
          "  LMax: 4\n"
          "  Mass: -1.0\n"
          "  Spin: [0.0, 1.0, 0.1]\n"
          "  Center: [-0.01, 0.02, 0.01]\n"
          "  InnerRadius: 1.0\n"
          "  OuterRadius: 10.0\n"),
      Catch::Contains("Tried to create a Shape TimeDependence, but the mass"));
}

}  // namespace
}  // namespace domain::creators::time_dependence
