// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace domain {

namespace {
template <size_t Dim>
void test_translation() {
  MAKE_GENERATOR(gen);
  // define vars for FunctionOfTime::PiecewisePolynomial f(t) = t**2.
  double t = -1.0;
  const double dt = 0.6;
  const double final_time = 4.0;
  constexpr size_t deriv_order = 3;

  const std::array<DataVector, deriv_order + 1> init_func{
      {{Dim, 1.0}, {Dim, -2.0}, {Dim, 2.0}, {Dim, 0.0}}};

  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  f_of_t_list["translation"] =
      std::make_unique<Polynomial>(t, init_func, final_time + dt);

  const FoftPtr& f_of_t = f_of_t_list.at("translation");

  const CoordinateMaps::TimeDependent::Translation<Dim> trans_map{
      "translation"};
  // test serialized/deserialized map
  const auto trans_map_deserialized = serialize_and_deserialize(trans_map);

  UniformCustomDistribution<double> dist_double{-5.0, 5.0};
  std::array<double, Dim> point_xi{};
  fill_with_random_values(make_not_null(&point_xi), make_not_null(&gen),
                          make_not_null(&dist_double));

  while (t < final_time) {
    std::array<double, Dim> translation{};
    for (size_t i = 0; i < Dim; i++) {
      gsl::at(translation, i) = square(t);
    }
    std::array<double, Dim> frame_vel{};
    for (size_t i = 0; i < Dim; i++) {
      gsl::at(frame_vel, i) = f_of_t->func_and_deriv(t)[1][i];
    }

    CHECK_ITERABLE_APPROX(trans_map(point_xi, t, f_of_t_list),
                          point_xi + translation);
    CHECK_ITERABLE_APPROX(
        trans_map.inverse(point_xi + translation, t, f_of_t_list).value(),
        point_xi);
    CHECK_ITERABLE_APPROX(trans_map.frame_velocity(point_xi, t, f_of_t_list),
                          frame_vel);

    CHECK_ITERABLE_APPROX(trans_map_deserialized(point_xi, t, f_of_t_list),
                          point_xi + translation);
    CHECK_ITERABLE_APPROX(
        trans_map_deserialized.inverse(point_xi + translation, t, f_of_t_list)
            .value(),
        point_xi);
    CHECK_ITERABLE_APPROX(trans_map_deserialized.frame_velocity(
                              point_xi + translation, t, f_of_t_list),
                          frame_vel);

    t += dt;
  }

  // time-independent checks
  {
    const auto identity_matrix = identity<Dim>(point_xi[0]);
    const auto jacobian = trans_map.jacobian(point_xi);
    const auto jacobian_deserialized =
        trans_map_deserialized.jacobian(point_xi);
    const auto inv_jacobian = trans_map.inv_jacobian(point_xi);
    const auto inv_jacobian_deserialized =
        trans_map_deserialized.inv_jacobian(point_xi);

    CHECK_ITERABLE_APPROX(jacobian, identity_matrix);
    CHECK_ITERABLE_APPROX(jacobian_deserialized, identity_matrix);
    CHECK_ITERABLE_APPROX(inv_jacobian, identity_matrix);
    CHECK_ITERABLE_APPROX(inv_jacobian_deserialized, identity_matrix);
  }

  // Check inequivalence operator
  CHECK_FALSE(trans_map != trans_map);
  CHECK_FALSE(trans_map_deserialized != trans_map_deserialized);

  // Check serialization
  CHECK(trans_map == trans_map_deserialized);
  CHECK_FALSE(trans_map != trans_map_deserialized);

  test_coordinate_map_argument_types(trans_map, point_xi, t, f_of_t_list);
  CHECK(not CoordinateMaps::TimeDependent::Translation<Dim>{}.is_identity());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.TimeDependent.Translation",
                  "[Domain][Unit]") {
  test_translation<1>();
  test_translation<2>();
  test_translation<3>();
}
}  // namespace domain
