// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/WithNoise.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<MathFunction<1, Frame::Inertial>, tmpl::list<>>,
        tmpl::pair<gh::BoundaryConditions::BoundaryCondition<Dim>,
                   tmpl::list<gh::BoundaryConditions::DirichletAnalytic<Dim>>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   gh::Solutions::all_solutions<Dim>>>;
  };
};

template <size_t Dim>
struct ConvertPlaneWave {
  using unpacked_container = int;
  using packed_container =
      gh::Solutions::WrappedGr<gr::Solutions::GaugeWave<Dim>>;
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
  register_classes_with_charm(gh::Solutions::all_solutions<Dim>{});
  MAKE_GENERATOR(gen);
  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time, Tags::AnalyticSolution<gh::Solutions::WrappedGr<
                      gr::Solutions::GaugeWave<Dim>>>>>(
      0.5, ConvertPlaneWave<Dim>::create_container());

  helpers::test_boundary_condition_with_python<
      gh::BoundaryConditions::DirichletAnalytic<Dim>,
      gh::BoundaryConditions::BoundaryCondition<Dim>, gh::System<Dim>,
      tmpl::list<gh::BoundaryCorrections::UpwindPenalty<Dim>>,
      tmpl::list<ConvertPlaneWave<Dim>>,
      tmpl::list<Tags::AnalyticSolution<
          gh::Solutions::WrappedGr<gr::Solutions::GaugeWave<Dim>>>>,
      Metavariables<Dim>>(
      make_not_null(&gen),
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions."
      "DirichletAnalytic",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<
              gr::Tags::SpacetimeMetric<DataVector, Dim>>,
          helpers::Tags::PythonFunctionName<gh::Tags::Pi<DataVector, Dim>>,
          helpers::Tags::PythonFunctionName<gh::Tags::Phi<DataVector, Dim>>,
          helpers::Tags::PythonFunctionName<gh::Tags::ConstraintGamma1>,
          helpers::Tags::PythonFunctionName<gh::Tags::ConstraintGamma2>,
          helpers::Tags::PythonFunctionName<gr::Tags::Lapse<DataVector>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Shift<DataVector, Dim>>>{
          "error", "spacetime_metric", "pi", "phi", "constraint_gamma1",
          "constraint_gamma2", "lapse", "shift"},
      "DirichletAnalytic:\n"
      "  AnalyticPrescription:\n"
      "    GeneralizedHarmonic(GaugeWave):\n"
      "      Amplitude: 0.2\n"
      "      Wavelength: 10.0\n",
      Index<Dim - 1>{Dim == 1 ? 1 : 5}, box_analytic_soln,
      tuples::TaggedTuple<helpers::Tags::Range<gh::Tags::ConstraintGamma1>,
                          helpers::Tags::Range<gh::Tags::ConstraintGamma2>>{
          std::array{0.0, 1.0}, std::array{0.0, 1.0}});
}
// Verify that DirichletAnalytic unwraps WithNoise and uses only the inner
// analytic solution for boundary values, so noise has no effect on the BC.
template <size_t Dim>
void test_with_noise_unwrapping() {
  CAPTURE(Dim);
  const size_t n_pts = 3;

  const double amplitude = 0.2;
  const double wavelength = 10.0;
  auto make_gauge_wave = [&]() {
    return std::make_unique<
        gh::Solutions::WrappedGr<gr::Solutions::GaugeWave<Dim>>>(amplitude,
                                                                 wavelength);
  };

  const gh::BoundaryConditions::DirichletAnalytic<Dim> bc_plain{
      make_gauge_wave()};
  const gh::BoundaryConditions::DirichletAnalytic<Dim> bc_with_noise{
      std::make_unique<evolution::initial_data::WithNoise>(
          make_gauge_wave(), 1.0, 132_st, std::vector<std::string>{"All"})};

  tnsr::I<DataVector, Dim, Frame::Inertial> coords{n_pts};
  for (size_t d = 0; d < Dim; ++d) {
    for (size_t i = 0; i < n_pts; ++i) {
      coords.get(d)[i] =
          0.1 * static_cast<double>(i + 1) + 0.05 * static_cast<double>(d);
    }
  }
  const tnsr::i<DataVector, Dim, Frame::Inertial> normal_covector{n_pts, 0.0};
  const tnsr::I<DataVector, Dim, Frame::Inertial> normal_vector{n_pts, 0.0};
  const Scalar<DataVector> interior_gamma1{DataVector(n_pts, 0.1)};
  const Scalar<DataVector> interior_gamma2{DataVector(n_pts, 0.5)};
  const double time = 0.5;

  tnsr::aa<DataVector, Dim> g_plain{n_pts};
  tnsr::aa<DataVector, Dim> pi_plain{n_pts};
  tnsr::iaa<DataVector, Dim> phi_plain{n_pts};
  Scalar<DataVector> lapse_plain{n_pts};
  Scalar<DataVector> gamma1_plain{n_pts};
  Scalar<DataVector> gamma2_plain{n_pts};
  tnsr::I<DataVector, Dim> shift_plain{n_pts};
  tnsr::II<DataVector, Dim> inv_spatial_plain{n_pts};
  bc_plain.dg_ghost(make_not_null(&g_plain), make_not_null(&pi_plain),
                    make_not_null(&phi_plain), make_not_null(&gamma1_plain),
                    make_not_null(&gamma2_plain), make_not_null(&lapse_plain),
                    make_not_null(&shift_plain),
                    make_not_null(&inv_spatial_plain), std::nullopt,
                    normal_covector, normal_vector, coords, interior_gamma1,
                    interior_gamma2, time);

  tnsr::aa<DataVector, Dim> g_noise{n_pts};
  tnsr::aa<DataVector, Dim> pi_noise{n_pts};
  tnsr::iaa<DataVector, Dim> phi_noise{n_pts};
  Scalar<DataVector> lapse_noise{n_pts};
  Scalar<DataVector> gamma1_noise{n_pts};
  Scalar<DataVector> gamma2_noise{n_pts};
  tnsr::I<DataVector, Dim> shift_noise{n_pts};
  tnsr::II<DataVector, Dim> inv_spatial_noise{n_pts};
  bc_with_noise.dg_ghost(
      make_not_null(&g_noise), make_not_null(&pi_noise),
      make_not_null(&phi_noise), make_not_null(&gamma1_noise),
      make_not_null(&gamma2_noise), make_not_null(&lapse_noise),
      make_not_null(&shift_noise), make_not_null(&inv_spatial_noise),
      std::nullopt, normal_covector, normal_vector, coords, interior_gamma1,
      interior_gamma2, time);

  // WithNoise is unwrapped for BCs: results must be identical to the plain BC
  CHECK_ITERABLE_APPROX(g_plain, g_noise);
  CHECK_ITERABLE_APPROX(pi_plain, pi_noise);
  CHECK_ITERABLE_APPROX(phi_plain, phi_noise);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GeneralizedHarmonic.BoundaryConditions.DirichletAnalytic",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test<1>();
  test<2>();
  test<3>();
  test_with_noise_unwrapping<1>();
  test_with_noise_unwrapping<2>();
  test_with_noise_unwrapping<3>();
}
