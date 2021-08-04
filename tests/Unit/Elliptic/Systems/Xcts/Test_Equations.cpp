// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Elliptic/FirstOrderSystem.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/FunctionInfo.hpp"

namespace {

void test_equations(const DataVector& used_for_size) {
  const double eps = 1.e-12;
  const auto seed = std::random_device{}();
  const double fill_result_tensors = 0.;
  pypp::check_with_random_values<3>(
      &Xcts::add_hamiltonian_sources<0>, "Equations", {"hamiltonian_sources"},
      {{{0., 1.}, {-1., 1.}, {0.5, 2.}}}, used_for_size, eps, seed,
      fill_result_tensors);
  pypp::check_with_random_values<3>(
      &Xcts::add_hamiltonian_sources<6>, "Equations",
      {"hamiltonian_sources_conf"}, {{{0., 1.}, {-1., 1.}, {0.5, 2.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<4>(
      &Xcts::add_linearized_hamiltonian_sources<0>, "Equations",
      {"linearized_hamiltonian_sources"},
      {{{0., 1.}, {-1., 1.}, {0.5, 2.}, {-1., 1.}}}, used_for_size, eps, seed,
      fill_result_tensors);
  pypp::check_with_random_values<4>(
      &Xcts::add_linearized_hamiltonian_sources<6>, "Equations",
      {"linearized_hamiltonian_sources_conf"},
      {{{0., 1.}, {-1., 1.}, {0.5, 2.}, {-1., 1.}}}, used_for_size, eps, seed,
      fill_result_tensors);
  pypp::check_with_random_values<2>(
      &Xcts::add_distortion_hamiltonian_sources, "Equations",
      {"distortion_hamiltonian_sources"}, {{{-1., 1.}, {0.5, 2.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<3>(
      &Xcts::add_linearized_distortion_hamiltonian_sources, "Equations",
      {"linearized_distortion_hamiltonian_sources"},
      {{{-1., 1.}, {0.5, 2.}, {-1., 1.}}}, used_for_size, eps, seed,
      fill_result_tensors);
  pypp::check_with_random_values<2>(
      &Xcts::add_curved_hamiltonian_or_lapse_sources, "Equations",
      {"curved_hamiltonian_or_lapse_sources"}, {{{-1., 1.}, {0.5, 2.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<7>(
      &Xcts::add_lapse_sources<0>, "Equations", {"lapse_sources"},
      {{{0., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<7>(
      &Xcts::add_lapse_sources<6>, "Equations", {"lapse_sources_conf"},
      {{{0., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<9>(&Xcts::add_linearized_lapse_sources<0>,
                                    "Equations", {"linearized_lapse_sources"},
                                    {{{0., 1.},
                                      {-1., 1.},
                                      {-1., 1.},
                                      {-1., 1.},
                                      {-1., 1.},
                                      {0.5, 2.},
                                      {0.5, 2.},
                                      {-1., 1.},
                                      {-1., 1.}}},
                                    used_for_size, eps, seed,
                                    fill_result_tensors);
  pypp::check_with_random_values<9>(
      &Xcts::add_linearized_lapse_sources<6>, "Equations",
      {"linearized_lapse_sources_conf"},
      {{{0., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<3>(
      &Xcts::add_distortion_hamiltonian_and_lapse_sources, "Equations",
      {"distortion_hamiltonian_sources_with_lapse", "distortion_lapse_sources"},
      {{{-1., 1.}, {0.5, 2.}, {0.5, 2.}}}, used_for_size, eps, seed,
      fill_result_tensors);
  pypp::check_with_random_values<5>(
      &Xcts::add_linearized_distortion_hamiltonian_and_lapse_sources,
      "Equations",
      {"linearized_distortion_hamiltonian_sources_with_lapse",
       "linearized_distortion_lapse_sources"},
      {{{-1., 1.}, {0.5, 2.}, {0.5, 2.}, {-1., 1.}, {-1., 1.}}}, used_for_size,
      eps, seed, fill_result_tensors);
  pypp::check_with_random_values<8>(
      &Xcts::add_flat_cartesian_momentum_sources<0>, "Equations",
      {"flat_cartesian_distortion_hamiltonian_sources_full",
       "flat_cartesian_distortion_lapse_sources_with_shift",
       "flat_cartesian_momentum_sources"},
      {{{-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<8>(
      &Xcts::add_flat_cartesian_momentum_sources<6>, "Equations",
      {"flat_cartesian_distortion_hamiltonian_sources_full",
       "flat_cartesian_distortion_lapse_sources_with_shift",
       "flat_cartesian_momentum_sources_conf"},
      {{{-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<10>(
      &Xcts::add_curved_momentum_sources<0>, "Equations",
      {"distortion_hamiltonian_sources_full",
       "distortion_lapse_sources_with_shift", "momentum_sources"},
      {{{-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<10>(
      &Xcts::add_curved_momentum_sources<6>, "Equations",
      {"distortion_hamiltonian_sources_full",
       "distortion_lapse_sources_with_shift", "momentum_sources_conf"},
      {{{-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<13>(
      &Xcts::add_flat_cartesian_linearized_momentum_sources<0>, "Equations",
      {"flat_cartesian_linearized_distortion_hamiltonian_sources_full",
       "flat_cartesian_linearized_distortion_lapse_sources_with_shift",
       "flat_cartesian_linearized_momentum_sources"},
      {{{-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<13>(
      &Xcts::add_flat_cartesian_linearized_momentum_sources<6>, "Equations",
      {"flat_cartesian_linearized_distortion_hamiltonian_sources_full",
       "flat_cartesian_linearized_distortion_lapse_sources_with_shift",
       "flat_cartesian_linearized_momentum_sources_conf"},
      {{{-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<15>(
      &Xcts::add_curved_linearized_momentum_sources<0>, "Equations",
      {"linearized_distortion_hamiltonian_sources_full",
       "linearized_distortion_lapse_sources_with_shift",
       "linearized_momentum_sources"},
      {{{-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<15>(
      &Xcts::add_curved_linearized_momentum_sources<6>, "Equations",
      {"linearized_distortion_hamiltonian_sources_full",
       "linearized_distortion_lapse_sources_with_shift",
       "linearized_momentum_sources_conf"},
      {{{-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {0.5, 2.},
        {0.5, 2.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.},
        {-1., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
}

template <Xcts::Equations EnabledEquations, Xcts::Geometry ConformalGeometry,
          int ConformalMatterScale>
void test_computers(const DataVector& used_for_size) {
  CAPTURE(EnabledEquations);
  CAPTURE(ConformalGeometry);
  CAPTURE(ConformalMatterScale);
  using system = Xcts::FirstOrderSystem<EnabledEquations, ConformalGeometry,
                                        ConformalMatterScale>;
  TestHelpers::elliptic::test_first_order_fluxes_computer<system>(
      used_for_size);
  TestHelpers::elliptic::test_first_order_sources_computer<system>(
      used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Xcts", "[Unit][Elliptic]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Elliptic/Systems/Xcts"};
  GENERATE_UNINITIALIZED_DATAVECTOR;
  test_equations(dv);
  CHECK_FOR_DATAVECTORS(
      test_computers,
      (Xcts::Equations::Hamiltonian, Xcts::Equations::HamiltonianAndLapse,
       Xcts::Equations::HamiltonianLapseAndShift),
      (Xcts::Geometry::FlatCartesian, Xcts::Geometry::Curved), (0, 6, 8));
}
