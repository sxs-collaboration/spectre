// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedVariables.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/DirichletCharacteristics.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/System.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

namespace helpers = TestHelpers::evolution::dg;

namespace {
struct CopyPsiFromInterior : db::SimpleTag {
  using type = bool;
};

struct ZeroIncomingMode : db::SimpleTag {
  using type = bool;
};

template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            SecondOrderScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
            SecondOrderScalarWave::BoundaryConditions::
                standard_boundary_conditions<Dim>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   SecondOrderScalarWave::Solutions::all_solutions<Dim>>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>>;
  };
};

template <size_t Dim>
void test() {
  using namespace std::string_literals;
  register_classes_with_charm(
      SecondOrderScalarWave::Solutions::all_solutions<Dim>{});
  register_classes_with_charm(
      MathFunctions::all_math_functions<1, Frame::Inertial>{});
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);

  using Psi = SecondOrderScalarWave::Tags::Psi;
  using Pi = SecondOrderScalarWave::Tags::Pi;
  using Phi = SecondOrderScalarWave::Tags::Phi<Dim>;
  using DtBoundaryPsi = ::Tags::dt<evolution::dg::Tags::BoundaryValue<Psi>>;

  const auto factory_string = [](const bool copy_psi_from_interior,
                                 const bool zero_incoming_mode) {
    const std::string plane_wave =
        "\n"
        "    SecondOrderPlaneWave:\n"
        "      WaveVector: [0.1" +
        (Dim > 1 ? std::string{", 1.1"} : std::string{}) +
        (Dim > 2 ? std::string{", 2.1"} : std::string{}) +
        "]\n"
        "      Center: [1.1" +
        (Dim > 1 ? std::string{", 0.1"} : std::string{}) +
        (Dim > 2 ? std::string{", -0.9"} : std::string{}) +
        "]\n"
        "      Profile:\n"
        "        Gaussian:\n"
        "          Amplitude: 0.9\n"
        "          Width: 0.6\n"
        "          Center: 0.0\n";
    return "DirichletCharacteristics:\n"
           "  AnalyticPrescription:" +
           (zero_incoming_mode ? " ZeroIncomingMode\n"s : plane_wave) +
           "  CopyPsiFromInterior: " +
           (copy_psi_from_interior ? "true\n"s : "false\n"s);
  };

  for (const auto& [copy_psi_from_interior, zero_incoming_mode] :
       {std::pair{false, false}, std::pair{true, false}, std::pair{false, true},
        std::pair{true, true}}) {
    CAPTURE(copy_psi_from_interior);
    CAPTURE(zero_incoming_mode);
    const auto box = db::create<
        db::AddSimpleTags<Tags::Time, CopyPsiFromInterior, ZeroIncomingMode>>(
        0.5, copy_psi_from_interior, zero_incoming_mode);
    helpers::test_boundary_condition_with_python<
        SecondOrderScalarWave::BoundaryConditions::DirichletCharacteristics<
            Dim>,
        SecondOrderScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
        SecondOrderScalarWave::System<Dim>,
        tmpl::list<
            SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>>,
        tmpl::list<>, tmpl::list<CopyPsiFromInterior, ZeroIncomingMode>,
        Metavariables<Dim>>(
        make_not_null(&gen),
        "Evolution.Systems.SecondOrderScalarWave.BoundaryConditions."
        "DirichletCharacteristics",
        tuples::TaggedTuple<
            helpers::Tags::PythonFunctionForErrorMessage<>,
            helpers::Tags::PythonFunctionForBoundaryFieldDtErrorMessage,
            helpers::Tags::PythonFunctionName<Psi>,
            helpers::Tags::PythonFunctionName<Pi>,
            helpers::Tags::PythonFunctionName<Phi>,
            helpers::Tags::PythonFunctionName<DtBoundaryPsi>>{
            "error", "dt_boundary_psi_error", "ghost_psi", "ghost_pi",
            "ghost_phi", "dt_boundary_psi"},
        factory_string(copy_psi_from_interior, zero_incoming_mode),
        Index<Dim - 1>{Dim == 1 ? 1 : 5}, box, tuples::TaggedTuple<>{});
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.SecondOrderScalarWave.BoundaryConditions.DirichletCharacteristics",
    "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{""};
  test<1>();
  test<2>();
  test<3>();
}
