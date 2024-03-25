// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/SelfForce.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {

void test_self_force_acceleration() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<double, 3> (*)(
          const Scalar<double>&, const tnsr::i<double, 3>&,
          const tnsr::I<double, 3>&, const double, const double,
          const tnsr::AA<double, 3>&, const Scalar<double>&)>(
          self_force_acceleration<3>),
      "SelfForce", "self_force_acceleration", {{{-2.0, 2.0}}}, 1);
}

void test_self_force_per_mass() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<double, 3> (*)(
          const tnsr::a<double, 3>&, const tnsr::A<double, 3>&, const double,
          const double, const tnsr::AA<double, 3>&)>(self_force_per_mass<3>),
      "SelfForce", "self_force_per_mass", {{{-2.0, 2.0}}}, 1);
}

void test_dt_self_force_per_mass() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<double, 3> (*)(
          const tnsr::a<double, 3>&, const tnsr::a<double, 3>&,
          const tnsr::A<double, 3>&, const tnsr::A<double, 3>&, const double,
          const double, const tnsr::AA<double, 3>&,
          const tnsr::AA<double, 3>&)>(dt_self_force_per_mass<3>),
      "SelfForce", "dt_self_force_per_mass", {{{-2.0, 2.0}}}, 1);
}

void test_dt2_self_force_per_mass() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<double, 3> (*)(
          const tnsr::a<double, 3>&, const tnsr::a<double, 3>&,
          const tnsr::a<double, 3>&, const tnsr::A<double, 3>&,
          const tnsr::A<double, 3>&, const tnsr::A<double, 3>&, const double,
          const double, const tnsr::AA<double, 3>&, const tnsr::AA<double, 3>&,
          const tnsr::AA<double, 3>&)>(dt2_self_force_per_mass<3>),
      "SelfForce", "dt2_self_force_per_mass", {{{-2.0, 2.0}}}, 1);
}

void test_Du_self_force_per_mass() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<double, 3> (*)(
          const tnsr::A<double, 3>&, const tnsr::A<double, 3>&,
          const tnsr::A<double, 3>&, const tnsr::Abb<double, 3>&)>(
          Du_self_force_per_mass<3>),
      "SelfForce", "Du_self_force_per_mass", {{{-2.0, 2.0}}}, 1);
}

void test_dt_Du_self_force_per_mass() {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<double, 3> (*)(
          const tnsr::A<double, 3>&, const tnsr::A<double, 3>&,
          const tnsr::A<double, 3>&, const tnsr::A<double, 3>&,
          const tnsr::A<double, 3>&, const tnsr::Abb<double, 3>&,
          const tnsr::Abb<double, 3>&)>(dt_Du_self_force_per_mass<3>),
      "SelfForce", "dt_Du_self_force_per_mass", {{{-2.0, 2.0}}}, 1);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CSW.Worldtube.SelfForce",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/CurvedScalarWave/Worldtube"};
  test_self_force_acceleration();
  test_self_force_per_mass();
  test_dt_self_force_per_mass();
  test_dt2_self_force_per_mass();
  test_Du_self_force_per_mass();
  test_dt_Du_self_force_per_mass();
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
