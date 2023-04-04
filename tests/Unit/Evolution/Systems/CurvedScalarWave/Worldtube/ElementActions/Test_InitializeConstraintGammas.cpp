// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/InitializeConstraintGammas.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <size_t Dim>
void test_initialize_constraint_damping_gammas(
    const gsl::not_null<std::mt19937*> gen,
    const gsl::not_null<std::uniform_real_distribution<>*> dist) {
  const size_t num_points = 100;
  const auto random_coords =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          gen, dist, DataVector(num_points));
  auto box = db::create<
      db::AddSimpleTags<CurvedScalarWave::Tags::ConstraintGamma1,
                        CurvedScalarWave::Tags::ConstraintGamma2,
                        domain::Tags::Coordinates<Dim, Frame::Inertial>>>(
      Scalar<DataVector>{}, Scalar<DataVector>{}, random_coords);
  db::mutate_apply<CurvedScalarWave::Worldtube::Initialization::
                       InitializeConstraintDampingGammas<Dim>>(
      make_not_null(&box));
  CHECK(get<CurvedScalarWave::Tags::ConstraintGamma1>(box) ==
        Scalar<DataVector>{num_points, 0.});
  CHECK_ITERABLE_APPROX(
      get(get<CurvedScalarWave::Tags::ConstraintGamma2>(box)),
      10. * exp(-square(get(magnitude(random_coords)) * 1.e-1)) + 1.e-4);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CSW.Worldtube.InitializeConstraintDampingGammas",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-10., 10.);
  test_initialize_constraint_damping_gammas<1>(make_not_null(&gen),
                                               make_not_null(&dist));
  test_initialize_constraint_damping_gammas<2>(make_not_null(&gen),
                                               make_not_null(&dist));
  test_initialize_constraint_damping_gammas<3>(make_not_null(&gen),
                                               make_not_null(&dist));
}
