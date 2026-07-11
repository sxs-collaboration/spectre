// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Hypertorus.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Literals.hpp"

namespace domain {
namespace {
template <size_t Dim>
struct HypertorusOptions;

template <>
struct HypertorusOptions<1> {
  static constexpr auto lower_bounds = std::array{-2.};
  static constexpr auto upper_bounds = std::array{3.};
  static constexpr auto max_modes = std::array{4_st};
  static constexpr auto grid_velocity = std::array{0.3};
  static constexpr auto name = "PeriodicInterval";
};

template <>
struct HypertorusOptions<2> {
  static constexpr auto lower_bounds = std::array{-2., -3.};
  static constexpr auto upper_bounds = std::array{3., 4.};
  static constexpr auto max_modes = std::array{4_st, 5_st};
  static constexpr auto grid_velocity = std::array{0.3, 0.4};
  static constexpr auto name = "PeriodicRectangle";
};

template <>
struct HypertorusOptions<3> {
  static constexpr auto lower_bounds = std::array{-2., -3., -4.};
  static constexpr auto upper_bounds = std::array{3., 4., 5.};
  static constexpr auto max_modes = std::array{4_st, 5_st, 6_st};
  static constexpr auto grid_velocity = std::array{0.3, 0.4, 0.5};
  static constexpr auto name = "PeriodicBrick";
};

template <typename T, size_t Dim>
std::string to_string(const std::array<T, Dim>& a) {
  std::string result = get_output(a).replace(0, 1, "[");
  return result.replace(result.length() - 1, 1, "]");
}

template <size_t Dim>
std::string option_string(const bool time_dependent) {
  const std::string time_dependence_option =
      time_dependent
          ? "  TimeDependence:\n"
            "    UniformTranslation:\n"
            "      InitialTime: 0.0\n"
            "      Velocity: " +
                to_string(HypertorusOptions<Dim>::grid_velocity) + "\n"
          : "  TimeDependence: None\n";
  return std::string(HypertorusOptions<Dim>::name) +
         ":\n"
         "  LowerBound: " +
         to_string(HypertorusOptions<Dim>::lower_bounds) +
         "\n"
         "  UpperBound: " +
         to_string(HypertorusOptions<Dim>::upper_bounds) +
         "\n"
         "  InitialMaximumModeNumber: " +
         to_string(HypertorusOptions<Dim>::max_modes) + "\n" +
         time_dependence_option;
}

template <size_t Dim>
std::vector<std::array<size_t, Dim>> expected_extents(
    const std::array<size_t, Dim>& max_modes) {
  if constexpr (Dim == 1) {
    return std::vector{std::array{2 * max_modes[0] + 1}};
  } else if constexpr (Dim == 2) {
    return std::vector{std::array{2 * max_modes[0] + 1, 2 * max_modes[1] + 1}};
  } else {
    return std::vector{std::array{2 * max_modes[0] + 1, 2 * max_modes[1] + 1,
                                  2 * max_modes[2] + 1}};
  }
}

template <size_t Dim, typename Generator>
void test_hypertorus_construction(
    const gsl::not_null<Generator*> gen,
    const domain::creators::Hypertorus<Dim>& creator,
    const std::array<double, Dim>& lower_bounds,
    const std::array<double, Dim>& upper_bounds,
    const std::array<size_t, Dim>& max_modes,
    const bool expect_boundary_conditions, const std::vector<double>& times,
    const std::array<double, Dim>& grid_velocity) {
  CHECK(creator.grid_anchors().empty());

  const auto all_boundary_conditions = creator.external_boundary_conditions();
  CHECK(all_boundary_conditions.size() == 1);
  CHECK(all_boundary_conditions[0].empty());

  CHECK(creator.initial_extents() == expected_extents(max_modes));
  CHECK(creator.initial_refinement_levels() ==
        std::vector{make_array<Dim>(0_st)});

  const auto block_names = creator.block_names();
  const auto& name = HypertorusOptions<Dim>::name;
  CHECK(block_names.size() == 1);
  CHECK(block_names[0] == name);

  const auto block_groups = creator.block_groups();
  CHECK(block_groups.size() == 1);
  REQUIRE(block_groups.contains(name));
  CHECK(block_groups.at(name).size() == 1);
  CHECK(block_groups.at(name).contains(name));

  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      creator, expect_boundary_conditions);
  CHECK(domain.excision_spheres().empty());

  const auto& blocks = domain.blocks();
  const size_t num_blocks = blocks.size();
  CHECK(num_blocks == 1);
  const auto& block = blocks[0];

  CHECK(block.external_boundaries().size() == 0);
  CHECK(block.topologies() == domain::topologies::hypertorus<Dim>);

  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  const ElementMap<Dim, Frame::Grid> grid_element_map{ElementId<Dim>{0}, block};
  const ElementMap<Dim, Frame::Inertial> inertial_element_map{ElementId<Dim>{0},
                                                              block};
  const double phi = phi_distribution(*gen);
  const tnsr::I<double, Dim, Frame::ElementLogical> x_logical{phi};
  CAPTURE(x_logical);
  const auto functions_of_time = creator.functions_of_time();
  constexpr double two_pi = 2. * std::numbers::pi;
  for (const double t : times) {
    CAPTURE(t);
    CAPTURE(grid_velocity);
    const auto x_grid = grid_element_map(x_logical, t, functions_of_time);
    CAPTURE(x_grid);
    const auto expected_x_grid = [&x_logical, &lower_bounds, &upper_bounds]() {
      tnsr::I<double, Dim, Frame::Grid> result;
      for (size_t d = 0; d < Dim; ++d) {
        const double scale =
            (gsl::at(upper_bounds, d) - gsl::at(lower_bounds, d)) / two_pi;
        result.get(d) = gsl::at(lower_bounds, d) + x_logical.get(d) * scale;
      }
      return result;
    }();
    CHECK_ITERABLE_APPROX(x_grid, expected_x_grid);
    const auto x_inertial =
        inertial_element_map(x_logical, t, functions_of_time);
    const auto expected_x_inertial = [&expected_x_grid, &t, &grid_velocity]() {
      tnsr::I<double, Dim, Frame::Inertial> result;
      for (size_t d = 0; d < Dim; ++d) {
        result.get(d) = expected_x_grid.get(d) + gsl::at(grid_velocity, d) * t;
      }
      return result;
    }();
    CHECK_ITERABLE_APPROX(x_inertial, expected_x_inertial);
  }
}

template <size_t Dim, typename Generator>
void test_hypertorus(const gsl::not_null<Generator*> gen) {
  const auto lower_bounds = HypertorusOptions<Dim>::lower_bounds;
  const auto upper_bounds = HypertorusOptions<Dim>::upper_bounds;
  const auto max_modes = HypertorusOptions<Dim>::max_modes;
  const auto grid_velocity = HypertorusOptions<Dim>::grid_velocity;
  for (const auto expect_bcs : {true, false}) {
    for (const auto time_dependent : {true, false}) {
      CAPTURE(time_dependent);
      const domain::creators::Hypertorus<Dim> creator{
          lower_bounds, upper_bounds, max_modes,
          time_dependent
              ? std::make_unique<
                    domain::creators::time_dependence::UniformTranslation<Dim>>(
                    0.0, grid_velocity)
              : nullptr};
      test_hypertorus_construction(
          gen, creator, lower_bounds, upper_bounds, max_modes, expect_bcs,
          std::vector{1.0, 10.0},
          time_dependent ? grid_velocity : make_array<Dim>(0.0));
      TestHelpers::domain::creators::test_creation(
          option_string<Dim>(time_dependent), creator, expect_bcs);
    }
  }
}

template <size_t Dim>
void test_parse_errors() {
  INFO("Test parse errors");
  const auto lower_bounds = make_array<Dim>(0.);
  const auto bad_bounds = make_array<Dim>(-1.);
  const auto max_modes = make_array<Dim>(1_st);

  CHECK_THROWS_WITH(
      creators::Hypertorus<Dim>(lower_bounds, bad_bounds, max_modes),
      Catch::Matchers::ContainsSubstring(
          "must be strictly smaller than upper bound"));
}

template <size_t Dim, typename Generator>
void test(const gsl::not_null<Generator*> gen) {
  test_hypertorus<Dim>(gen);
  test_parse_errors<Dim>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Hypertorus", "[Domain][Unit]") {
  MAKE_GENERATOR(gen);
  domain::creators::time_dependence::register_derived_with_charm();
  test<1>(make_not_null(&gen));
  test<2>(make_not_null(&gen));
  test<3>(make_not_null(&gen));
}
}  // namespace domain
