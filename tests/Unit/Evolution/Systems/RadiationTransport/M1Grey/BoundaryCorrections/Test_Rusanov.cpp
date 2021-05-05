// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

void test(const gsl::not_null<std::mt19937*> gen, const size_t num_pts) {
  using neutrino_species = tmpl::list<neutrinos::ElectronNeutrinos<1>,
                                      neutrinos::ElectronAntiNeutrinos<1>>;
  using system = RadiationTransport::M1Grey::System<neutrino_species>;
  using rusanov = RadiationTransport::M1Grey::BoundaryCorrections::Rusanov<
      neutrino_species>;

  helpers::test_boundary_correction_conservation<system>(
      gen, rusanov{},
      Mesh<2>{num_pts, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      {}, {});

  helpers::test_boundary_correction_with_python<system>(
      gen, "Rusanov",
      {{"dg_package_data_tilde_e_nue", "dg_package_data_tilde_e_bar_nue",
        "dg_package_data_tilde_s_nue", "dg_package_data_tilde_s_bar_nue",
        "dg_package_data_normal_dot_flux_tilde_e_nue",
        "dg_package_data_normal_dot_flux_tilde_e_bar_nue",
        "dg_package_data_normal_dot_flux_tilde_s_nue",
        "dg_package_data_normal_dot_flux_tilde_s_bar_nue"}},
      {{"dg_boundary_terms_tilde_e_nue", "dg_boundary_terms_tilde_e_bar_nue",
        "dg_boundary_terms_tilde_s_nue", "dg_boundary_terms_tilde_s_bar_nue"}},
      rusanov{},
      Mesh<2>{num_pts, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      {}, {});

  const auto rusanov_from_factory = TestHelpers::test_creation<
      std::unique_ptr<RadiationTransport::M1Grey::BoundaryCorrections::
                          BoundaryCorrection<neutrino_species>>>("Rusanov:");

  helpers::test_boundary_correction_with_python<system>(
      gen, "Rusanov",
      {{"dg_package_data_tilde_e_nue", "dg_package_data_tilde_e_bar_nue",
        "dg_package_data_tilde_s_nue", "dg_package_data_tilde_s_bar_nue",
        "dg_package_data_normal_dot_flux_tilde_e_nue",
        "dg_package_data_normal_dot_flux_tilde_e_bar_nue",
        "dg_package_data_normal_dot_flux_tilde_s_nue",
        "dg_package_data_normal_dot_flux_tilde_s_bar_nue"}},
      {{"dg_boundary_terms_tilde_e_nue", "dg_boundary_terms_tilde_e_bar_nue",
        "dg_boundary_terms_tilde_s_nue", "dg_boundary_terms_tilde_s_bar_nue"}},
      dynamic_cast<const rusanov&>(*rusanov_from_factory),
      Mesh<2>{num_pts, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      {}, {});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RadiationTransport.M1Grey.Rusanov",
                  "[Unit][Evolution]") {
  using neutrino_species = tmpl::list<neutrinos::ElectronNeutrinos<1>,
                                      neutrinos::ElectronAntiNeutrinos<1>>;
  using rusanov = RadiationTransport::M1Grey::BoundaryCorrections::Rusanov<
      neutrino_species>;
  PUPable_reg(rusanov);
  Parallel::register_derived_classes_with_charm<rusanov>();

  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RadiationTransport/M1Grey/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  test(make_not_null(&gen), 5);
}
