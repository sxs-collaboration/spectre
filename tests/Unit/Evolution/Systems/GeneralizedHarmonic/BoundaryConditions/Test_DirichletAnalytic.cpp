// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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
  using packed_container =
      GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::GaugeWave<Dim>>;
  using packed_type = double;

  static packed_container create_container() {
    const double amplitude = 0.2;
    const double wavelength = 10.0;
    return {amplitude, wavelength};
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
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time,
      Tags::AnalyticSolution<GeneralizedHarmonic::Solutions::WrappedGr<
          gr::Solutions::GaugeWave<Dim>>>>>(
      0.5, ConvertPlaneWave<Dim>::create_container());

  helpers::test_boundary_condition_with_python<
      GeneralizedHarmonic::BoundaryConditions::DirichletAnalytic<Dim>,
      GeneralizedHarmonic::BoundaryConditions::BoundaryCondition<Dim>,
      GeneralizedHarmonic::System<Dim>,
      tmpl::list<GeneralizedHarmonic::BoundaryCorrections::UpwindPenalty<Dim>>,
      tmpl::list<ConvertPlaneWave<Dim>>>(
      make_not_null(&gen),
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions."
      "DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<gr::Tags::SpacetimeMetric<Dim>>,
          helpers::Tags::PythonFunctionName<GeneralizedHarmonic::Tags::Pi<Dim>>,
          helpers::Tags::PythonFunctionName<
              GeneralizedHarmonic::Tags::Phi<Dim>>,
          helpers::Tags::PythonFunctionName<
              GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>,
          helpers::Tags::PythonFunctionName<
              GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>,
          helpers::Tags::PythonFunctionName<gr::Tags::Lapse<DataVector>>,
          helpers::Tags::PythonFunctionName<
              gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>>{
          "error", "spacetime_metric", "pi", "phi", "constraint_gamma1",
          "constraint_gamma2", "lapse", "shift"},
      "DirichletAnalytic:\n", Index<Dim - 1>{Dim == 1 ? 1 : 5},
      box_analytic_soln,
      tuples::TaggedTuple<
          helpers::Tags::Range<
              GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>,
          helpers::Tags::Range<
              GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>>{
          std::array{0.0, 1.0}, std::array{0.0, 1.0}});
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GeneralizedHarmonic.BoundaryConditions.DirichletAnalytic",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test<1>();
  test<2>();
  test<3>();
}
