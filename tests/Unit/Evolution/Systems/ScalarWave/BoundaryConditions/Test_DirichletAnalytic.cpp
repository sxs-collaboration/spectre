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
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
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

  static inline unpacked_container unpack(
      const packed_container& /*packed*/,
      const size_t /*grid_point_index*/) noexcept {
    // No way of getting the args from the boundary condition.
    return Dim;
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

template <size_t Dim>
void test() {
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
      tmpl::list<ConvertPlaneWave<Dim>>>(
      make_not_null(&gen), "DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<ScalarWave::Psi>,
          helpers::Tags::PythonFunctionName<ScalarWave::Pi>,
          helpers::Tags::PythonFunctionName<ScalarWave::Phi<Dim>>,
          helpers::Tags::PythonFunctionName<
              ScalarWave::Tags::ConstraintGamma2>>{"error", "psi", "pi", "phi",
                                                   "constraint_gamma2"},
      "DirichletAnalytic:\n", Index<Dim - 1>{Dim == 1 ? 1 : 5},
      box_analytic_soln,
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
