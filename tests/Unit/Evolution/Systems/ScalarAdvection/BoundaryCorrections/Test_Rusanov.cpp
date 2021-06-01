// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/ScalarAdvection/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {

template <size_t Dim>
void test_rusanov(const gsl::not_null<std::mt19937*> gen,
                  const size_t num_pts) {
  helpers::test_boundary_correction_conservation<ScalarAdvection::System<Dim>>(
      gen, ScalarAdvection::BoundaryCorrections::Rusanov<Dim>{},
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      {}, {});

  helpers::test_boundary_correction_with_python<ScalarAdvection::System<Dim>>(
      gen, "Rusanov",
      {{"dg_package_data_u", "dg_package_data_normal_dot_flux_u",
        "dg_package_data_abs_char_speed"}},
      {{"dg_boundary_terms_u"}},
      ScalarAdvection::BoundaryCorrections::Rusanov<Dim>{},
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      {}, {});

  const auto rusanov = TestHelpers::test_creation<std::unique_ptr<
      ScalarAdvection::BoundaryCorrections::BoundaryCorrection<Dim>>>(
      "Rusanov:");

  helpers::test_boundary_correction_with_python<ScalarAdvection::System<Dim>>(
      gen, "Rusanov",
      {{"dg_package_data_u", "dg_package_data_normal_dot_flux_u",
        "dg_package_data_abs_char_speed"}},
      {{"dg_boundary_terms_u"}},
      dynamic_cast<const ScalarAdvection::BoundaryCorrections::Rusanov<Dim>&>(
          *rusanov),
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      {}, {});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ScalarAdvection.BoundaryCorrections.Rusanov",
                  "[Unit][Evolution]") {
  PUPable_reg(ScalarAdvection::BoundaryCorrections::Rusanov<1>);
  PUPable_reg(ScalarAdvection::BoundaryCorrections::Rusanov<2>);

  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarAdvection/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  test_rusanov<1>(make_not_null(&gen), 5);
  test_rusanov<2>(make_not_null(&gen), 5);
}
