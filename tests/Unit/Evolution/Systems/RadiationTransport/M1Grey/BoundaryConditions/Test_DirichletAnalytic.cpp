// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryCorrections/Rusanov.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticData/RadiationTransport/M1Grey/Factory.hpp"
#include "PointwiseFunctions/AnalyticData/RadiationTransport/M1Grey/HomogeneousSphere.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/ConstantM1.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {

// compare to other test_dirichlet analytic data
template <typename NeutrinoSpeciesList>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using initial_data_list =
        tmpl::append<RadiationTransport::M1Grey::AnalyticData::all_data,
                     RadiationTransport::M1Grey::Solutions::all_solutions>;
    using factory_classes = tmpl::map<
        tmpl::pair<RadiationTransport::M1Grey::BoundaryConditions::
                       BoundaryCondition<NeutrinoSpeciesList>,
                   tmpl::list<RadiationTransport::M1Grey::BoundaryConditions::
                                  DirichletAnalytic<NeutrinoSpeciesList>>>,
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>>;
  };
};

struct ConvertConstantM1 {
  using unpacked_container = int;
  using packed_container = RadiationTransport::M1Grey::Solutions::ConstantM1;
  using packed_type = double;

  static packed_container create_container() {
    const std::array<double, 3> mean_velocity_{{0.1, 0.2, 0.3}};
    const double comoving_energy_density = 0.4;
    return {mean_velocity_, comoving_energy_density};
  }

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    // No way of getting the args from the boundary condition.
    return 3;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    *packed = create_container();
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

struct ConvertHomogeneousSphere {
  using unpacked_container = int;
  using packed_container =
      RadiationTransport::M1Grey::AnalyticData::HomogeneousSphere;
  using packed_type = double;

  static packed_container create_container() {
    const double radius = 1.0;
    const double emissivity_and_opacity = 1.0;
    const double outer_opacity = 0.5;
    const double boundary_roundness = 0.03;
    return {radius, emissivity_and_opacity, outer_opacity, boundary_roundness};
  }

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    // No way of getting the args from the boundary condition.
    return 3;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    *packed = create_container();
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

void test() {
  using initial_data_list =
      tmpl::append<RadiationTransport::M1Grey::AnalyticData::all_data,
                   RadiationTransport::M1Grey::Solutions::all_solutions>;
  register_classes_with_charm(initial_data_list{});

  MAKE_GENERATOR(gen);

  // Databox for constant M1
  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time, Tags::AnalyticSolution<
                      RadiationTransport::M1Grey::Solutions::ConstantM1>>>(
      0.5, ConvertConstantM1::create_container());

  // Databox for homogeneous sphere
  const auto box_analytic_soln_homogen_sphere = db::create<db::AddSimpleTags<
      Tags::Time, Tags::AnalyticData<RadiationTransport::M1Grey::AnalyticData::
                                         HomogeneousSphere>>>(
      0.5, ConvertHomogeneousSphere::create_container());

  using neutrino_species = tmpl::list<neutrinos::ElectronNeutrinos<1>,
                                      neutrinos::ElectronAntiNeutrinos<1>>;
  using system = RadiationTransport::M1Grey::System<neutrino_species>;
  using boundary_condition =
      RadiationTransport::M1Grey::BoundaryConditions::BoundaryCondition<
          neutrino_species>;
  using dirichlet_analytic =
      RadiationTransport::M1Grey::BoundaryConditions::DirichletAnalytic<
          neutrino_species>;
  using rusanov = RadiationTransport::M1Grey::BoundaryCorrections::Rusanov<
      neutrino_species>;

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

  // Constant M1
  helpers::test_boundary_condition_with_python<
      dirichlet_analytic, boundary_condition, system, tmpl::list<rusanov>,
      tmpl::list<ConvertConstantM1>,
      tmpl::list<Tags::AnalyticSolution<
          RadiationTransport::M1Grey::Solutions::ConstantM1>>,
      Metavariables<neutrino_species>>(
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
          // python functions labeled below
          "soln_error", "soln_tilde_e_nue_const", "soln_tilde_e_bar_nue_const",
          "soln_tilde_s_nue_const", "soln_tilde_s_bar_nue_const",
          "soln_flux_tilde_e_nue_const", "soln_flux_tilde_e_bar_nue_const",
          "soln_flux_tilde_s_nue_const", "soln_flux_tilde_s_bar_nue_const"},
      "DirichletAnalytic:\n"
      "  AnalyticPrescription:\n"
      "    ConstantM1:\n"
      "      MeanVelocity: [0.1, 0.2, 0.3]\n"
      "      ComovingEnergyDensity: 0.4\n",
      Index<2>{5}, box_analytic_soln, tuples::TaggedTuple<>{});

  // Homogeneous sphere
  helpers::test_boundary_condition_with_python<
      dirichlet_analytic, boundary_condition, system, tmpl::list<rusanov>,
      tmpl::list<ConvertHomogeneousSphere>,
      tmpl::list<Tags::AnalyticData<
          RadiationTransport::M1Grey::AnalyticData::HomogeneousSphere>>,
      Metavariables<neutrino_species>>(
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
          // python functions labeled below
          "soln_error", "soln_tilde_e_nue", "soln_tilde_e_bar_nue",
          "soln_tilde_s_nue", "soln_tilde_s_bar_nue", "soln_flux_tilde_e_nue",
          "soln_flux_tilde_e_bar_nue", "soln_flux_tilde_s_nue",
          "soln_flux_tilde_s_bar_nue"},
      "DirichletAnalytic:\n"
      "  AnalyticPrescription:\n"
      "    HomogeneousSphere:\n"
      "      Radius: 1.0\n"
      "      EmissivityAndOpacity: 1.0\n"
      "      OuterOpacity: 0.5\n"
      "      BoundaryRoundness: 0.03\n", Index<2>{5},
      box_analytic_soln_homogen_sphere, tuples::TaggedTuple<>{});
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.RadiationTransport.M1Grey.BoundaryConditions.DirichletAnalytic",
    "[Unit][Evolution]") {
  using neutrino_species = tmpl::list<neutrinos::ElectronNeutrinos<1>,
                                      neutrinos::ElectronAntiNeutrinos<1>>;

  using dirichlet_analytic =
      RadiationTransport::M1Grey::BoundaryConditions::DirichletAnalytic<
          neutrino_species>;
  PUPable_reg(dirichlet_analytic);

  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test();
}
