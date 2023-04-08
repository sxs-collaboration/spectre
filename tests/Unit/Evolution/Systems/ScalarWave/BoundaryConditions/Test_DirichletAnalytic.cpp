// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/Factory.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            ScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
            tmpl::list<ScalarWave::BoundaryConditions::DirichletAnalytic<Dim>>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   ScalarWave::Solutions::all_solutions<Dim>>,
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   MathFunctions::all_math_functions<1, Frame::Inertial>>>;
  };
};

template <size_t Dim>
struct ConvertPlaneWave {
  using unpacked_container = int;
  using packed_container = ScalarWave::Solutions::PlaneWave<Dim>;
  using packed_type = double;

  static packed_container create_container() {
    std::array<double, Dim> wave_vector{};
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(wave_vector, i) = 0.1 + i;
    }
    std::array<double, Dim> center{};
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(center, i) = 1.1 - i;
    }
    return {wave_vector, center,
            std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
                0.9, 0.6, 0.0)};
  }

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    // No way of getting the args from the boundary condition.
    return Dim;
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

template <size_t Dim>
void test() {
  register_classes_with_charm(ScalarWave::Solutions::all_solutions<Dim>{});
  register_classes_with_charm(
      MathFunctions::all_math_functions<1, Frame::Inertial>{});
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time,
      Tags::AnalyticSolution<ScalarWave::Solutions::PlaneWave<Dim>>>>(
      0.5, ConvertPlaneWave<Dim>::create_container());

  helpers::test_boundary_condition_with_python<
      ScalarWave::BoundaryConditions::DirichletAnalytic<Dim>,
      ScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
      ScalarWave::System<Dim>,
      tmpl::list<ScalarWave::BoundaryCorrections::UpwindPenalty<Dim>>,
      tmpl::list<ConvertPlaneWave<Dim>>,
      tmpl::list<Tags::AnalyticSolution<ScalarWave::Solutions::PlaneWave<Dim>>>,
      Metavariables<Dim>>(
      make_not_null(&gen), "DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<ScalarWave::Tags::Psi>,
          helpers::Tags::PythonFunctionName<ScalarWave::Tags::Pi>,
          helpers::Tags::PythonFunctionName<ScalarWave::Tags::Phi<Dim>>,
          helpers::Tags::PythonFunctionName<
              ScalarWave::Tags::ConstraintGamma2>>{"error", "psi", "pi", "phi",
                                                   "constraint_gamma2"},
      "DirichletAnalytic:\n"
      "  AnalyticPrescription:\n"
      "    PlaneWave:\n"
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
      Index<Dim - 1>{Dim == 1 ? 1 : 5}, box_analytic_soln,
      tuples::TaggedTuple<
          helpers::Tags::Range<ScalarWave::Tags::ConstraintGamma2>>{
          std::array{0.0, 1.0}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ScalarWave.BoundaryConditions.DirichletAnalytic",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarWave/BoundaryConditions/"};
  test<1>();
  test<2>();
  test<3>();
}
