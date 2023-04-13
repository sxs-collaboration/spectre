// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/SphereTimeDependentMaps.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame

namespace domain::creators::sphere {
namespace {
void test() {
  const double initial_time = 1.5;
  const std::array<double, 3> size_values{0.9, 0.08, 0.007};
  const size_t l_max = 8;

  TimeDependentMapOptions time_dep_options{initial_time, size_values, l_max};

  std::unordered_map<std::string, double> expiration_times{
      {TimeDependentMapOptions::shape_name, 15.5},
      {TimeDependentMapOptions::size_name,
       std::numeric_limits<double>::infinity()}};

  // These are hard-coded so this is just a regression test
  CHECK(TimeDependentMapOptions::size_name == "Size"s);
  CHECK(TimeDependentMapOptions::shape_name == "Shape"s);

  using PP2 = domain::FunctionsOfTime::PiecewisePolynomial<2>;
  using PP3 = domain::FunctionsOfTime::PiecewisePolynomial<3>;
  PP3 size{initial_time,
           std::array<DataVector, 4>{{{gsl::at(size_values, 0)},
                                      {gsl::at(size_values, 1)},
                                      {gsl::at(size_values, 2)},
                                      {0.0}}},
           expiration_times.at(TimeDependentMapOptions::size_name)};
  const DataVector shape_zeros{YlmSpherepack::spectral_size(l_max, l_max), 0.0};
  PP2 shape{initial_time,
            std::array<DataVector, 3>{shape_zeros, shape_zeros, shape_zeros},
            expiration_times.at(TimeDependentMapOptions::shape_name)};

  const auto functions_of_time =
      time_dep_options.create_functions_of_time(expiration_times);

  CHECK(dynamic_cast<PP3&>(
            *functions_of_time.at(TimeDependentMapOptions::size_name).get()) ==
        size);
  CHECK(dynamic_cast<PP2&>(
            *functions_of_time.at(TimeDependentMapOptions::shape_name).get()) ==
        shape);

  const std::array<double, 3> center{-5.0, -0.01, -0.02};
  const double inner_radius = 0.5;
  const double outer_radius = 2.1;

  for (const bool include_distorted : make_array(true, false)) {
    time_dep_options.build_maps(center, inner_radius, outer_radius);

    const auto grid_to_distorted_map =
        time_dep_options.grid_to_distorted_map(include_distorted);
    const auto grid_to_inertial_map =
        time_dep_options.grid_to_inertial_map(include_distorted);
    const auto distorted_to_inertial_map =
        time_dep_options.distorted_to_inertial_map(include_distorted);

    // All of these maps are tested individually. Rather than going through the
    // effort of coming up with a source coordinate and calculating analytically
    // what we would get after it's mapped, we just check that whether it's
    // supposed to be a nullptr and if it's not, if it's the identity and that
    // the jacobians are time dependent.
    const auto check_map = [](const auto& map, const bool is_null,
                              const bool is_identity) {
      if (is_null) {
        CHECK(map == nullptr);
      } else {
        CHECK(map->is_identity() == is_identity);
        CHECK(map->inv_jacobian_is_time_dependent() == not is_identity);
        CHECK(map->jacobian_is_time_dependent() == not is_identity);
      }
    };

    check_map(grid_to_inertial_map, false, not include_distorted);
    check_map(grid_to_distorted_map, not include_distorted, false);
    check_map(distorted_to_inertial_map, not include_distorted, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.SphereTimeDependentMaps",
                  "[Domain][Unit]") {
  test();
}
}  // namespace domain::creators::sphere
