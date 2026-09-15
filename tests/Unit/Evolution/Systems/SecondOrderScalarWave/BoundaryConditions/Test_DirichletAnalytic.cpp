// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/System.hpp"
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
#include "Utilities/TMPL.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
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
  register_classes_with_charm(
      SecondOrderScalarWave::Solutions::all_solutions<Dim>{});
  register_classes_with_charm(
      MathFunctions::all_math_functions<1, Frame::Inertial>{});
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  const auto box = db::create<db::AddSimpleTags<Tags::Time>>(0.5);

  helpers::test_boundary_condition_with_python<
      SecondOrderScalarWave::BoundaryConditions::DirichletAnalytic<Dim>,
      SecondOrderScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
      SecondOrderScalarWave::System<Dim>,
      tmpl::list<
          SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>>,
      tmpl::list<>, tmpl::list<>, Metavariables<Dim>>(
      make_not_null(&gen), "DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<SecondOrderScalarWave::Tags::Psi>,
          helpers::Tags::PythonFunctionName<SecondOrderScalarWave::Tags::Pi>,
          helpers::Tags::PythonFunctionName<
              SecondOrderScalarWave::Tags::Phi<Dim>>>{"error", "psi", "pi",
                                                      "phi"},
      "DirichletAnalytic:\n"
      "  AnalyticPrescription:\n"
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
          "          Center: 0.0\n",
      Index<Dim - 1>{Dim == 1 ? 1 : 5}, box, tuples::TaggedTuple<>{});
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.SecondOrderScalarWave.BoundaryConditions.DirichletAnalytic",
    "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/"};
  test<1>();
  test<2>();
  test<3>();
}
