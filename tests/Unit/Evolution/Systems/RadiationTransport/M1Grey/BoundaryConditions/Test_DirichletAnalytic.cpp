// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/ConstantM1.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
struct ConvertConstantM1 {
  using unpacked_container = int;
  using packed_container = RadiationTransport::M1Grey::Solutions::ConstantM1;
  using packed_type = double;

  static packed_container create_container() {
    const std::array<double, 3> mean_velocity_{{0.1, 0.2, 0.3}};
    const double comoving_energy_density = 0.4;
    return {mean_velocity_, comoving_energy_density};
  }

  static inline unpacked_container unpack(
      const packed_container& /*packed*/,
      const size_t /*grid_point_index*/) noexcept {
    // No way of getting the args from the boundary condition.
    return 3;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    *packed = create_container();
  }

  static inline size_t get_size(const packed_container& /*packed*/) noexcept {
    return 1;
  }
};

void test() {
  MAKE_GENERATOR(gen);
  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time, Tags::AnalyticSolution<
                      RadiationTransport::M1Grey::Solutions::ConstantM1>>>(
      0.5, ConvertConstantM1::create_container());

  using system = RadiationTransport::M1Grey::System<tmpl::list<
      neutrinos::ElectronNeutrinos<1>, neutrinos::ElectronAntiNeutrinos<1>>>;
  using boundary_condition =
      RadiationTransport::M1Grey::BoundaryConditions::BoundaryCondition<
          neutrinos::ElectronNeutrinos<1>, neutrinos::ElectronAntiNeutrinos<1>>;
  using dirichlet_analytic =
      RadiationTransport::M1Grey::BoundaryConditions::DirichletAnalytic<
          neutrinos::ElectronNeutrinos<1>, neutrinos::ElectronAntiNeutrinos<1>>;
  using rusanov = RadiationTransport::M1Grey::BoundaryCorrections::Rusanov<
      neutrinos::ElectronNeutrinos<1>, neutrinos::ElectronAntiNeutrinos<1>>;

  using tilde_e_nue_tag =
      RadiationTransport::M1Grey::Tags::TildeE<Frame::Inertial,
                                               neutrinos::ElectronNeutrinos<1>>;
  using tilde_e_bar_nue_tag = RadiationTransport::M1Grey::Tags::TildeE<
      Frame::Inertial, neutrinos::ElectronAntiNeutrinos<1>>;
  using tilde_s_nue_tag =
      RadiationTransport::M1Grey::Tags::TildeS<Frame::Inertial,
                                               neutrinos::ElectronNeutrinos<1>>;
  using tilde_s_bar_nue_tag = RadiationTransport::M1Grey::Tags::TildeS<
      Frame::Inertial, neutrinos::ElectronAntiNeutrinos<1>>;

  helpers::test_boundary_condition_with_python<
      dirichlet_analytic, boundary_condition, system, tmpl::list<rusanov>,
      tmpl::list<ConvertConstantM1>>(
      make_not_null(&gen),
      "Evolution.Systems.RadiationTransport.M1Grey.BoundaryConditions."
      "DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<tilde_e_nue_tag>,
          helpers::Tags::PythonFunctionName<tilde_e_bar_nue_tag>,
          helpers::Tags::PythonFunctionName<tilde_s_nue_tag>,
          helpers::Tags::PythonFunctionName<tilde_s_bar_nue_tag>,

          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<tilde_e_nue_tag, tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              tilde_e_bar_nue_tag, tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::Flux<tilde_s_nue_tag, tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              tilde_s_bar_nue_tag, tmpl::size_t<3>, Frame::Inertial>>>{
          "soln_error", "soln_tilde_e_nue", "soln_tilde_e_bar_nue",
          "soln_tilde_s_nue", "soln_tilde_s_bar_nue", "soln_flux_tilde_e_nue",
          "soln_flux_tilde_e_bar_nue", "soln_flux_tilde_s_nue",
          "soln_flux_tilde_s_bar_nue"},
      "DirichletAnalytic:\n", Index<2>{5}, box_analytic_soln,
      tuples::TaggedTuple<>{});

  // Note: Currently there is no analytic data for the
  // RadiationTransport::M1Grey system. When that is implemented, a test should
  // be added.
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.RadiationTransport.M1Grey.BoundaryConditions.DirichletAnalytic",
    "[Unit][Evolution]") {
  using dirichlet_analytic =
      RadiationTransport::M1Grey::BoundaryConditions::DirichletAnalytic<
          neutrinos::ElectronNeutrinos<1>, neutrinos::ElectronAntiNeutrinos<1>>;
  PUPable_reg(dirichlet_analytic);
  Parallel::register_derived_classes_with_charm<dirichlet_analytic>();

  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test();
}
