// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <string>

#include "Evolution/Systems/ScalarWave/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen, const size_t num_pts) {
  PUPable_reg(ScalarWave::BoundaryCorrections::UpwindPenalty<Dim>);
  TestHelpers::evolution::dg::test_boundary_correction_conservation<
      ScalarWave::System<Dim>>(
      gen, ScalarWave::BoundaryCorrections::UpwindPenalty<Dim>{},
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      {}, {});

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      ScalarWave::System<Dim>>(
      gen, "UpwindPenalty",
      {{"dg_package_data_char_speed_v_psi", "dg_package_data_char_speed_v_zero",
        "dg_package_data_char_speed_v_plus",
        "dg_package_data_char_speed_v_minus",
        "dg_package_data_char_speed_v_plus_times_normal",
        "dg_package_data_char_speed_v_minus_times_normal",
        "dg_package_data_char_speed_gamma2_v_psi",
        "dg_package_data_char_speeds"}},
      {{"dg_boundary_terms_pi", "dg_boundary_terms_phi",
        "dg_boundary_terms_psi"}},
      ScalarWave::BoundaryCorrections::UpwindPenalty<Dim>{},
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      {}, {});

  const auto upwind_penalty = TestHelpers::test_factory_creation<
      ScalarWave::BoundaryCorrections::BoundaryCorrection<Dim>>(
      "UpwindPenalty:");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      ScalarWave::System<Dim>>(
      gen, "UpwindPenalty",
      {{"dg_package_data_char_speed_v_psi", "dg_package_data_char_speed_v_zero",
        "dg_package_data_char_speed_v_plus",
        "dg_package_data_char_speed_v_minus",
        "dg_package_data_char_speed_v_plus_times_normal",
        "dg_package_data_char_speed_v_minus_times_normal",
        "dg_package_data_char_speed_gamma2_v_psi",
        "dg_package_data_char_speeds"}},
      {{"dg_boundary_terms_pi", "dg_boundary_terms_phi",
        "dg_boundary_terms_psi"}},
      dynamic_cast<const ScalarWave::BoundaryCorrections::UpwindPenalty<Dim>&>(
          *upwind_penalty),
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      {}, {});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ScalarWave.UpwindPenalty", "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarWave/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  test<1>(make_not_null(&gen), 1);
  test<2>(make_not_null(&gen), 5);
  test<3>(make_not_null(&gen), 5);
}
