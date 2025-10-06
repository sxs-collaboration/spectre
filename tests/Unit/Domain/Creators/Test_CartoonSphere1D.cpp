// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/CartoonSphere1D.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Options/Context.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition(const bool outer) {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      outer ? Direction<3>::upper_xi() : Direction<3>::lower_xi(), 50);
}

template <typename T>
std::string stringize(const std::vector<T>& t) {
  std::string result = get_output(t).replace(0, 1, "[");
  return result.replace(result.length() - 1, 1, "]");
}

std::string option_string(
    const double inner_radial_bound, const double outer_radial_bound,
    const size_t radial_refinement, const size_t radial_extents,
    const std::vector<double>& radial_partitioning,
    const std::vector<domain::CoordinateMaps::Distribution>&
        radial_distribution,
    const bool time_dependent, const bool with_boundary_conditions) {
  const std::string time_dependent_option =
      time_dependent ? "  TimeDependence:\n"
                       "    UniformTranslation:\n"
                       "      InitialTime: 1.0\n"
                       "      Velocity: [2.3, -0.3, 0.5]\n"
                     : "  TimeDependence: None\n";
  const std::string inner_bc_option = with_boundary_conditions
                                          ? "  InnerBoundaryCondition:\n"
                                            "    TestBoundaryCondition:\n"
                                            "      Direction: lower-xi\n"
                                            "      BlockId: 50\n"
                                          : "";
  const std::string outer_bc_option = with_boundary_conditions
                                          ? "  OuterBoundaryCondition:\n"
                                            "    TestBoundaryCondition:\n"
                                            "      Direction: upper-xi\n"
                                            "      BlockId: 50\n"
                                          : "";
  return "CartoonSphere1D:\n"
         "  InnerRadius: " +
         std::to_string(inner_radial_bound) +
         "\n"
         "  OuterRadius: " +
         std::to_string(outer_radial_bound) + "\n" +
         "  InitialRadialRefinement: " + std::to_string(radial_refinement) +
         "\n"
         "  InitialNumberOfRadialGridPoints: " +
         std::to_string(radial_extents) +
         "\n"
         "  RadialPartitioning: " +
         stringize(radial_partitioning) +
         "\n"
         "  RadialDistributions: " +
         (radial_distribution.size() == 1 ? get_output(radial_distribution[0])
                                          : stringize(radial_distribution)) +
         "\n" + time_dependent_option + inner_bc_option + outer_bc_option;
}

void test_parse_errors() {
  INFO("CartoonSphere1D check throws");
  const double lower_bound = 1.0;
  const double upper_bound = 2.0;
  const size_t radial_refinement = 2;
  const std::vector<size_t> radial_refinement_high = {{1, 1}};
  const size_t radial_extents = 5;
  const std::vector<size_t> radial_extents_high = {{2, 3}};
  const std::vector<double> radial_partitioning = {};
  const std::vector<double> radial_partitioning_unordered = {
      {1.5 * lower_bound, 1.1 * lower_bound}};
  const std::vector<double> radial_partitioning_low = {
      {0.5 * lower_bound, 1.1 * lower_bound}};
  const std::vector<double> radial_partitioning_high = {
      {2.1 * upper_bound, 2.2 * upper_bound}};
  const std::vector<domain::CoordinateMaps::Distribution> radial_distribution{
      domain::CoordinateMaps::Distribution::Linear};
  const std::vector<domain::CoordinateMaps::Distribution>
      radial_distribution_too_many{
          domain::CoordinateMaps::Distribution::Linear,
          domain::CoordinateMaps::Distribution::Logarithmic};

  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, 0.5 * lower_bound, radial_refinement, radial_extents,
          radial_partitioning, radial_distribution, nullptr, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Inner radius must be smaller than outer radius"));

  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, upper_bound, radial_refinement, radial_extents,
          radial_partitioning_unordered, radial_distribution, nullptr, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify radial partitioning in ascending order."));

  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, upper_bound, radial_refinement, radial_extents,
          radial_partitioning_low, radial_distribution, nullptr, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "First radial partition must be larger than the inner"));
  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, upper_bound, radial_refinement, radial_extents,
          radial_partitioning_high, radial_distribution, nullptr, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Last radial partition must be smaller than the outer"));
  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, upper_bound, radial_refinement, radial_extents,
          radial_partitioning, radial_distribution_too_many, nullptr, nullptr,
          nullptr, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify a 'RadialDistribution' for every spherical shell. You"));
  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, upper_bound, radial_refinement_high, radial_extents,
          radial_partitioning, radial_distribution, nullptr, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "must be the same size as RadialDistributions "));
  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, upper_bound, radial_refinement, radial_extents_high,
          radial_partitioning, radial_distribution, nullptr, nullptr, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "must be the same size as RadialDistributions "));
  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, upper_bound, radial_refinement, radial_extents,
          radial_partitioning, radial_distribution, nullptr,
          create_boundary_condition(false), nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Must specify either both inner and outer boundary conditions "
          "or neither."));
  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, upper_bound, radial_refinement, radial_extents,
          radial_partitioning, radial_distribution, nullptr,
          create_boundary_condition(false),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with CartoonSphere1D"));
  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, upper_bound, radial_refinement, radial_extents,
          radial_partitioning, radial_distribution, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          create_boundary_condition(true), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Cannot have periodic boundary conditions with CartoonSphere1D"));
  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, upper_bound, radial_refinement, radial_extents,
          radial_partitioning, radial_distribution, nullptr,
          create_boundary_condition(false),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      domain::creators::CartoonSphere1D(
          lower_bound, upper_bound, radial_refinement, radial_extents,
          radial_partitioning, radial_distribution, nullptr,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          create_boundary_condition(true), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like "
          "an outflow-type boundary condition, you must use that."));
}

void test_cartoon_sphere_construction(
    const domain::creators::CartoonSphere1D& cartoon_sphere,
    const double lower_bound, const double upper_bound,
    const std::vector<double>& radial_partitioning = {},
    const bool expect_boundary_conditions = true,
    const std::vector<double>& times = {0.},
    const std::array<double, 3>& velocity = {{0., 0., 0.}}) {
  // check consistency of domain
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      cartoon_sphere, expect_boundary_conditions, false, times);

  const auto& blocks = domain.blocks();
  const auto block_names = cartoon_sphere.block_names();
  const size_t num_blocks = blocks.size();
  CAPTURE(num_blocks);
  const auto all_boundary_conditions =
      cartoon_sphere.external_boundary_conditions();
  const auto functions_of_time = cartoon_sphere.functions_of_time();

  // Check total number of external boundaries
  const size_t num_shells = radial_partitioning.size() + 1;
  CHECK(num_blocks == num_shells);
  const size_t num_external_boundaries =
      alg::accumulate(blocks, 0_st, [](const size_t count, const auto& block) {
        return count + block.external_boundaries().size();
      });
  CHECK(num_external_boundaries == 2);

  std::vector<double> expected_radii = radial_partitioning;
  expected_radii.insert(expected_radii.begin(), lower_bound);
  expected_radii.emplace_back(upper_bound);

  for (size_t block_id = 0; block_id < num_blocks; ++block_id) {
    CAPTURE(block_id);
    const auto& block = blocks[block_id];
    const ElementMap<3, Frame::Inertial> inertial_element_map{
        ElementId<3>{block_id}, block};
    {
      INFO("Radius of random point on lower face");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{{{-1.0, 0, 0}}};
      for (const double current_time : times) {
        CAPTURE(current_time);
        auto x_inertial =
            inertial_element_map(x_logical, current_time, functions_of_time);
        const double delta_t = current_time - 1.0;
        for (size_t i = 0; i < 3; ++i) {
          x_inertial.get(i) -= gsl::at(velocity, i) * delta_t;
        }
        CHECK(get(magnitude(x_inertial)) == approx(expected_radii[block_id]));
      }
    }
    {
      INFO("Radius of random point on upper face");
      const tnsr::I<double, 3, Frame::ElementLogical> x_logical{{{1.0, 0, 0}}};
      for (const double current_time : times) {
        CAPTURE(current_time);
        auto x_inertial =
            inertial_element_map(x_logical, current_time, functions_of_time);
        const double delta_t = current_time - 1.0;
        for (size_t i = 0; i < 3; ++i) {
          x_inertial.get(i) -= gsl::at(velocity, i) * delta_t;
        }
        CHECK(get(magnitude(x_inertial)) ==
              approx(expected_radii[block_id + 1]));
      }
    }
    {
      INFO("External boundaries");
      const auto& external_boundaries = block.external_boundaries();
      if (num_blocks == 1) {
        CHECK(external_boundaries.size() == 2);
        CHECK(alg::found(external_boundaries, Direction<3>::lower_xi()));
        CHECK(alg::found(external_boundaries, Direction<3>::upper_xi()));
      } else if (block_id == 0) {
        CHECK(external_boundaries.size() == 1);
        CHECK(alg::found(external_boundaries, Direction<3>::lower_xi()));
      } else if (block_id == num_blocks - 1) {
        CHECK(external_boundaries.size() == 1);
        CHECK(alg::found(external_boundaries, Direction<3>::upper_xi()));
      } else {
        CHECK(external_boundaries.empty());
      }
    }
    if (expect_boundary_conditions) {
      INFO("Boundary conditions");
      const auto& boundary_conditions = all_boundary_conditions[block_id];
      for (const auto& direction : block.external_boundaries()) {
        CAPTURE(direction);
        const auto& boundary_condition =
            dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                             TestBoundaryCondition<3>&>(
                *boundary_conditions.at(direction));
        CHECK(boundary_condition.direction() == direction);
      }
    }
  }
}

template <typename Generator>
void test_sphere(const gsl::not_null<Generator*> gen) {
  const double lower_bound = 1.0;
  const double upper_bound = 2.0;
  const size_t radial_refinement = 3;
  const size_t radial_extents = 5;
  const double upper_minus_lower = upper_bound - lower_bound;
  const std::array<std::vector<double>, 3> radial_partitioning{
      {{},
       {0.5 * (lower_bound + upper_bound)},
       {lower_bound + 0.3 * upper_minus_lower,
        lower_bound + 0.6 * upper_minus_lower}}};
  const std::array<std::vector<domain::CoordinateMaps::Distribution>, 3>
      radial_distributions{{{domain::CoordinateMaps::Distribution::Linear},
                            {domain::CoordinateMaps::Distribution::Linear,
                             domain::CoordinateMaps::Distribution::Logarithmic},
                            {domain::CoordinateMaps::Distribution::Linear}}};

  const std::array<double, 3> velocity{{2.3, -0.3, 0.5}};
  const std::vector<double> times{1., 10.};
  for (auto [index, time_dependent, with_boundary_conditions] :
       random_sample<5>(
           cartesian_product(make_array(0_st, 1_st, 2_st),
                             make_array(true, false), make_array(true, false)),
           gen)) {
    CAPTURE(time_dependent);
    CAPTURE(with_boundary_conditions);
    CAPTURE(gsl::at(radial_partitioning, index));
    CAPTURE(gsl::at(radial_distributions, index));
    domain::creators::CartoonSphere1D::RadialDistributions::type
        radial_distributions_variant;
    if (gsl::at(radial_distributions, index).size() == 1) {
      radial_distributions_variant = gsl::at(radial_distributions, index)[0];
    } else {
      radial_distributions_variant = gsl::at(radial_distributions, index);
    }

    domain::creators::CartoonSphere1D::TimeDependence::type time_dependency{};

    auto translation_velocity = std::array<double, 3>{{0., 0., 0.}};

    if (time_dependent) {
      time_dependency = std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3, 0>>(
          1.0, velocity);
      translation_velocity = velocity;
    }

    const domain::creators::CartoonSphere1D cartoon_sphere{
        lower_bound,
        upper_bound,
        radial_refinement,
        radial_extents,
        gsl::at(radial_partitioning, index),
        radial_distributions_variant,
        std::move(time_dependency),
        with_boundary_conditions ? create_boundary_condition(false) : nullptr,
        with_boundary_conditions ? create_boundary_condition(true) : nullptr};
    test_cartoon_sphere_construction(
        cartoon_sphere, lower_bound, upper_bound,
        gsl::at(radial_partitioning, index), with_boundary_conditions,
        time_dependent ? times : std::vector<double>{1.}, translation_velocity);
    TestHelpers::domain::creators::test_creation(
        option_string(lower_bound, upper_bound, radial_refinement,
                      radial_extents, gsl::at(radial_partitioning, index),
                      gsl::at(radial_distributions, index), time_dependent,
                      with_boundary_conditions),
        cartoon_sphere, with_boundary_conditions);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.CartoonSphere1D", "[Domain][Unit]") {
  MAKE_GENERATOR(gen);
  domain::creators::time_dependence::register_derived_with_charm();
  test_parse_errors();
  test_sphere(make_not_null(&gen));
}
