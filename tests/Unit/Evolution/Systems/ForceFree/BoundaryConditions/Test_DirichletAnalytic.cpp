// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/ForceFree/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/Factory.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {

struct ConvertFfeBreakdown {
  using unpacked_container = double;
  using packed_container = ForceFree::AnalyticData::FfeBreakdown;
  using packed_type = double;

  static packed_container create_container() { return {}; }

  static inline unpacked_container unpack(const packed_container /*packed*/,
                                          const size_t /*grid_point_index*/) {
    return 0.0;  // no parameter but we need some placeholder type
  }

  static inline void pack(const gsl::not_null<packed_container*> /*packed*/,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    // no parameters but we need a placeholder function
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

struct ConvertFastwave {
  using unpacked_container = int;
  using packed_container = ForceFree::Solutions::FastWave;
  using packed_type = double;

  static packed_container create_container() { return {}; }

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    // No way of getting the args from the boundary condition.
    return 3;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    *packed = packed_container{};
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            ForceFree::BoundaryConditions::BoundaryCondition,
            tmpl::list<ForceFree::BoundaryConditions::DirichletAnalytic>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::list<ForceFree::Solutions::FastWave,
                              ForceFree::AnalyticData::FfeBreakdown>>>;
  };
};

void test_solution() {
  register_classes_with_charm(ForceFree::Solutions::all_solutions{});

  MAKE_GENERATOR(gen);

  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time, Tags::AnalyticSolution<ForceFree::Solutions::FastWave>,
      ForceFree::Tags::ParallelConductivity>>(
      0.5, ConvertFastwave::create_container(), 100.0);

  helpers::test_boundary_condition_with_python<
      ForceFree::BoundaryConditions::DirichletAnalytic,
      ForceFree::BoundaryConditions::BoundaryCondition, ForceFree::System,
      tmpl::list<ForceFree::BoundaryCorrections::Rusanov>,
      tmpl::list<ConvertFastwave>,
      tmpl::list<Tags::AnalyticSolution<ForceFree::Solutions::FastWave>>,
      Metavariables>(
      make_not_null(&gen),
      "Evolution.Systems.ForceFree.BoundaryConditions.DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,

          helpers::Tags::PythonFunctionName<ForceFree::Tags::TildeE>,
          helpers::Tags::PythonFunctionName<ForceFree::Tags::TildeB>,
          helpers::Tags::PythonFunctionName<ForceFree::Tags::TildePsi>,
          helpers::Tags::PythonFunctionName<ForceFree::Tags::TildePhi>,
          helpers::Tags::PythonFunctionName<ForceFree::Tags::TildeQ>,

          helpers::Tags::PythonFunctionName<::Tags::Flux<
              ForceFree::Tags::TildeE, tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              ForceFree::Tags::TildeB, tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              ForceFree::Tags::TildePsi, tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              ForceFree::Tags::TildePhi, tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              ForceFree::Tags::TildeQ, tmpl::size_t<3>, Frame::Inertial>>,

          helpers::Tags::PythonFunctionName<gr::Tags::Lapse<DataVector>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Shift<DataVector, 3>>>{
          "soln_error", "soln_tilde_e", "soln_tilde_b", "soln_tilde_psi",
          "soln_tilde_phi", "soln_tilde_q", "soln_flux_tilde_e",
          "soln_flux_tilde_b", "soln_flux_tilde_psi", "soln_flux_tilde_phi",
          "soln_flux_tilde_q", "soln_lapse", "soln_shift"},
      "DirichletAnalytic:\n"
      "  AnalyticPrescription:\n"
      "    FastWave:\n",
      Index<2>{5}, box_analytic_soln, tuples::TaggedTuple<>{});
}

void test_data() {
  MAKE_GENERATOR(gen);

  register_classes_with_charm(ForceFree::AnalyticData::all_data{});

  const auto box_analytic_data = db::create<db::AddSimpleTags<
      Tags::Time, Tags::AnalyticData<ForceFree::AnalyticData::FfeBreakdown>,
      ForceFree::Tags::ParallelConductivity>>(
      0.5, ConvertFfeBreakdown::create_container(), 100.0);

  helpers::test_boundary_condition_with_python<
      ForceFree::BoundaryConditions::DirichletAnalytic,
      ForceFree::BoundaryConditions::BoundaryCondition, ForceFree::System,
      tmpl::list<ForceFree::BoundaryCorrections::Rusanov>,
      tmpl::list<ConvertFfeBreakdown>,
      tmpl::list<Tags::AnalyticData<ForceFree::AnalyticData::FfeBreakdown>>,
      Metavariables>(
      make_not_null(&gen),
      "Evolution.Systems.ForceFree.BoundaryConditions.DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,

          helpers::Tags::PythonFunctionName<ForceFree::Tags::TildeE>,
          helpers::Tags::PythonFunctionName<ForceFree::Tags::TildeB>,
          helpers::Tags::PythonFunctionName<ForceFree::Tags::TildePsi>,
          helpers::Tags::PythonFunctionName<ForceFree::Tags::TildePhi>,
          helpers::Tags::PythonFunctionName<ForceFree::Tags::TildeQ>,

          helpers::Tags::PythonFunctionName<::Tags::Flux<
              ForceFree::Tags::TildeE, tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              ForceFree::Tags::TildeB, tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              ForceFree::Tags::TildePsi, tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              ForceFree::Tags::TildePhi, tmpl::size_t<3>, Frame::Inertial>>,
          helpers::Tags::PythonFunctionName<::Tags::Flux<
              ForceFree::Tags::TildeQ, tmpl::size_t<3>, Frame::Inertial>>,

          helpers::Tags::PythonFunctionName<gr::Tags::Lapse<DataVector>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Shift<DataVector, 3>>>{
          "soln_error", "data_tilde_e", "data_tilde_b", "data_tilde_psi",
          "data_tilde_phi", "data_tilde_q", "data_flux_tilde_e",
          "data_flux_tilde_b", "data_flux_tilde_psi", "data_flux_tilde_phi",
          "data_flux_tilde_q", "soln_lapse", "soln_shift"},
      "DirichletAnalytic:\n"
      "  AnalyticPrescription:\n"
      "    FfeBreakdown:\n",
      Index<2>{5}, box_analytic_data, tuples::TaggedTuple<>{});
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ForceFree.BoundaryConditions.DirichletAnalytic",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test_solution();
  test_data();
}
