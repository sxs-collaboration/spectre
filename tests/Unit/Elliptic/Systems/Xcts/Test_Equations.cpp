// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>

#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Utilities/TMPL.hpp"

namespace {

void test_equations(const DataVector& used_for_size) {
  const double eps = 1.e-12;
  const auto seed = std::random_device{}();
  const double fill_result_tensors = 0.;
  pypp::check_with_random_values<1>(
      &Xcts::add_hamiltonian_sources<0>, "Equations", {"hamiltonian_sources"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(&Xcts::add_hamiltonian_sources<6>,
                                    "Equations", {"hamiltonian_sources_conf"},
                                    {{{-1., 1.}}}, used_for_size, eps, seed,
                                    fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_linearized_hamiltonian_sources<0>, "Equations",
      {"linearized_hamiltonian_sources"}, {{{-1., 1.}}}, used_for_size, eps,
      seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_linearized_hamiltonian_sources<6>, "Equations",
      {"linearized_hamiltonian_sources_conf"}, {{{-1., 1.}}}, used_for_size,
      eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_distortion_hamiltonian_sources, "Equations",
      {"distortion_hamiltonian_sources"}, {{{-1., 1.}}}, used_for_size, eps,
      seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_linearized_distortion_hamiltonian_sources, "Equations",
      {"linearized_distortion_hamiltonian_sources"}, {{{-1., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_curved_hamiltonian_or_lapse_sources, "Equations",
      {"curved_hamiltonian_or_lapse_sources"}, {{{-1., 1.}}}, used_for_size,
      eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_lapse_sources<0>, "Equations", {"lapse_sources"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_lapse_sources<6>, "Equations", {"lapse_sources_conf"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(&Xcts::add_linearized_lapse_sources<0>,
                                    "Equations", {"linearized_lapse_sources"},
                                    {{{-1., 1.}}}, used_for_size, eps, seed,
                                    fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_linearized_lapse_sources<6>, "Equations",
      {"linearized_lapse_sources_conf"}, {{{-1., 1.}}}, used_for_size, eps,
      seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_distortion_hamiltonian_and_lapse_sources, "Equations",
      {"distortion_hamiltonian_sources_with_lapse", "distortion_lapse_sources"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_linearized_distortion_hamiltonian_and_lapse_sources,
      "Equations",
      {"linearized_distortion_hamiltonian_sources_with_lapse",
       "linearized_distortion_lapse_sources"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_flat_cartesian_momentum_sources<0>, "Equations",
      {"flat_cartesian_distortion_hamiltonian_sources_full",
       "flat_cartesian_distortion_lapse_sources_with_shift",
       "flat_cartesian_momentum_sources"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_flat_cartesian_momentum_sources<6>, "Equations",
      {"flat_cartesian_distortion_hamiltonian_sources_full",
       "flat_cartesian_distortion_lapse_sources_with_shift",
       "flat_cartesian_momentum_sources_conf"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_curved_momentum_sources<0>, "Equations",
      {"distortion_hamiltonian_sources_full",
       "distortion_lapse_sources_with_shift", "momentum_sources"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_curved_momentum_sources<6>, "Equations",
      {"distortion_hamiltonian_sources_full",
       "distortion_lapse_sources_with_shift", "momentum_sources_conf"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_flat_cartesian_linearized_momentum_sources<0>, "Equations",
      {"flat_cartesian_linearized_distortion_hamiltonian_sources_full",
       "flat_cartesian_linearized_distortion_lapse_sources_with_shift",
       "flat_cartesian_linearized_momentum_sources"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_flat_cartesian_linearized_momentum_sources<6>, "Equations",
      {"flat_cartesian_linearized_distortion_hamiltonian_sources_full",
       "flat_cartesian_linearized_distortion_lapse_sources_with_shift",
       "flat_cartesian_linearized_momentum_sources_conf"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_curved_linearized_momentum_sources<0>, "Equations",
      {"linearized_distortion_hamiltonian_sources_full",
       "linearized_distortion_lapse_sources_with_shift",
       "linearized_momentum_sources"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &Xcts::add_curved_linearized_momentum_sources<6>, "Equations",
      {"linearized_distortion_hamiltonian_sources_full",
       "linearized_distortion_lapse_sources_with_shift",
       "linearized_momentum_sources_conf"},
      {{{-1., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Xcts", "[Unit][Elliptic]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Elliptic/Systems/Xcts"};
  DataVector used_for_size{5};
  test_equations(used_for_size);
}
