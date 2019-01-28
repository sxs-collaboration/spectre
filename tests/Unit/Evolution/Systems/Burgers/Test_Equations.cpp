// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Burgers/Equations.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
DataVector apply_numerical_flux(const DataVector& ndotf_interior,
                                const DataVector& u_interior,
                                const DataVector& ndotf_exterior,
                                const DataVector& u_exterior) noexcept {
  using flux = Burgers::LocalLaxFriedrichsFlux;

  auto package_data_interior =
      make_with_value<Variables<typename flux::package_tags>>(u_interior, 0.);
  flux{}.package_data(&package_data_interior,
                      Scalar<DataVector>(ndotf_interior),
                      Scalar<DataVector>(u_interior));
  auto package_data_exterior =
      make_with_value<Variables<typename flux::package_tags>>(u_interior, 0.);
  flux{}.package_data(&package_data_exterior,
                      Scalar<DataVector>(ndotf_exterior),
                      Scalar<DataVector>(u_exterior));

  auto result = make_with_value<Scalar<DataVector>>(u_interior, 0.);
  flux{}(&result,
         get<Tags::NormalDotFlux<Burgers::Tags::U>>(package_data_interior),
         get<Burgers::Tags::U>(package_data_interior),
         get<Tags::NormalDotFlux<Burgers::Tags::U>>(package_data_exterior),
         get<Burgers::Tags::U>(package_data_exterior));
  return get(result);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Burgers.LocalLaxFriedrichsFlux", "[Unit][Burgers]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> interval(-10., 10.);
  const DataVector size(5);
  // Check general properties of fluxes
  {
    const auto u = make_with_random_values<DataVector>(
        make_not_null(&gen), make_not_null(&interval), size);
    const auto ndotf = make_with_random_values<DataVector>(
        make_not_null(&gen), make_not_null(&interval), size);
    CHECK_ITERABLE_APPROX(apply_numerical_flux(ndotf, u, -ndotf, u), ndotf);
  }
  {
    const auto u1 = make_with_random_values<DataVector>(
        make_not_null(&gen), make_not_null(&interval), size);
    const auto u2 = make_with_random_values<DataVector>(
        make_not_null(&gen), make_not_null(&interval), size);
    const auto ndotf1 = make_with_random_values<DataVector>(
        make_not_null(&gen), make_not_null(&interval), size);
    const auto ndotf2 = make_with_random_values<DataVector>(
        make_not_null(&gen), make_not_null(&interval), size);
    CHECK_ITERABLE_APPROX(apply_numerical_flux(ndotf1, u1, ndotf2, u2),
                          -apply_numerical_flux(ndotf2, u2, ndotf1, u1));
  }
}

SPECTRE_TEST_CASE("Unit.Burgers.ComputeLargestCharacteristicSpeed",
                  "[Unit][Burgers]") {
  CHECK(Burgers::ComputeLargestCharacteristicSpeed::apply(
            Scalar<DataVector>{{{{1., 2., 4., 3.}}}}) == 4.);
  CHECK(Burgers::ComputeLargestCharacteristicSpeed::apply(
            Scalar<DataVector>{{{{1., 2., 4., -5.}}}}) == 5.);
}
