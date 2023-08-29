// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen, const size_t num_pts) {
  PUPable_reg(gh::BoundaryCorrections::UpwindPenalty<Dim>);

  TestHelpers::evolution::dg::test_boundary_correction_conservation<
      gh::System<Dim>>(
      gen, gh::BoundaryCorrections::UpwindPenalty<Dim>{},
      Mesh<Dim - 1>{num_pts, SpatialDiscretization::Basis::Legendre,
                    SpatialDiscretization::Quadrature::Gauss},
      {}, {});

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      gh::System<Dim>>(
      gen, "UpwindPenalty",
      {{"dg_package_data_char_speed_v_spacetime_metric",
        "dg_package_data_char_speed_v_zero",
        "dg_package_data_char_speed_v_plus",
        "dg_package_data_char_speed_v_minus",
        "dg_package_data_char_speed_v_plus_times_normal",
        "dg_package_data_char_speed_v_minus_times_normal",
        "dg_package_data_char_speed_gamma2_v_spacetime_metric",
        "dg_package_data_char_speeds"}},
      {{"dg_boundary_terms_spacetime_metric", "dg_boundary_terms_pi",
        "dg_boundary_terms_phi"}},
      gh::BoundaryCorrections::UpwindPenalty<Dim>{},
      Mesh<Dim - 1>{num_pts, SpatialDiscretization::Basis::Legendre,
                    SpatialDiscretization::Quadrature::Gauss},
      {}, {});

  const auto upwind_penalty = TestHelpers::test_creation<
      std::unique_ptr<gh::BoundaryCorrections::BoundaryCorrection<Dim>>>(
      "UpwindPenalty:");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      gh::System<Dim>>(
      gen, "UpwindPenalty",
      {{"dg_package_data_char_speed_v_spacetime_metric",
        "dg_package_data_char_speed_v_zero",
        "dg_package_data_char_speed_v_plus",
        "dg_package_data_char_speed_v_minus",
        "dg_package_data_char_speed_v_plus_times_normal",
        "dg_package_data_char_speed_v_minus_times_normal",
        "dg_package_data_char_speed_gamma2_v_spacetime_metric",
        "dg_package_data_char_speeds"}},
      {{"dg_boundary_terms_spacetime_metric", "dg_boundary_terms_pi",
        "dg_boundary_terms_phi"}},
      dynamic_cast<const gh::BoundaryCorrections::UpwindPenalty<Dim>&>(
          *upwind_penalty),
      Mesh<Dim - 1>{num_pts, SpatialDiscretization::Basis::Legendre,
                    SpatialDiscretization::Quadrature::Gauss},
      {}, {});

  CHECK_FALSE(gh::BoundaryCorrections::UpwindPenalty<Dim>{} !=
              gh::BoundaryCorrections::UpwindPenalty<Dim>{});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.GeneralizedHarmonic.BoundaryCorrections.UpwindPenalty",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  test<1>(make_not_null(&gen), 1);
  test<2>(make_not_null(&gen), 5);
  test<3>(make_not_null(&gen), 5);
}
