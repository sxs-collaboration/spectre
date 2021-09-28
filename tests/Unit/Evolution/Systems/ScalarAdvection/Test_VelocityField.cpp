// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/VelocityField.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <size_t Dim>
void test_velocity_field(const gsl::not_null<std::mt19937*> gen) {
  // test for tags
  TestHelpers::db::test_compute_tag<
      ScalarAdvection::Tags::VelocityFieldCompute<Dim>>("VelocityField");

  // generate random coordinates
  const DataVector used_for_size(10);
  std::uniform_real_distribution<> distribution(0.0);
  if constexpr (Dim == 1) {
    // computation domain is [-1, 1] for 1D advection
    distribution = std::uniform_real_distribution<>(-1.0, 1.0);
  } else if constexpr (Dim == 2) {
    // computation domain is [0, 1] x [0, 1] for 2D advection
    distribution = std::uniform_real_distribution<>(0.0, 1.0);
  }
  const auto inertial_coords =
      make_with_random_values<tnsr::I<DataVector, Dim>>(
          gen, make_not_null(&distribution), used_for_size);

  // compute velocity field from VelocityFieldCompute struct
  auto velocity_field = make_with_value<tnsr::I<DataVector, Dim>>(
      inertial_coords, std::numeric_limits<double>::signaling_NaN());
  ScalarAdvection::Tags::VelocityFieldCompute<Dim>::function(&velocity_field,
                                                             inertial_coords);

  // compute velocity field from python implementation
  const tnsr::I<DataVector, Dim, Frame::Inertial> velocity_field_test{
      pypp::call<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          "TestFunctions", "velocity_field", inertial_coords)};

  // check values
  CHECK_ITERABLE_APPROX(velocity_field, velocity_field_test);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarAdvection.VelocityField",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarAdvection"};
  MAKE_GENERATOR(gen);

  test_velocity_field<1>(make_not_null(&gen));
  test_velocity_field<2>(make_not_null(&gen));
}
