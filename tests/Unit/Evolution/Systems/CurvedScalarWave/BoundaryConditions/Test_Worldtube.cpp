// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/Worldtube.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

// here we check that the dg_ghost field returns evolved variables which
// correspond to `VMinus` being calculated by the worldtube solution and all
// other fields calculated by the interior solution.
void test_dg_ghost(const gsl::not_null<std::mt19937*> gen) {
  static constexpr size_t Dim = 3;
  std::uniform_real_distribution<> dist(-10., 10.);
  gr::Solutions::KerrSchild kerr_schild{2., {0.1, 0.2, 0.3}, {0.4, 0.6, 0.9}};
  const size_t num_points = 1000.;
  const auto coords =
      make_with_random_values<tnsr::I<DataVector, 3>>(gen, dist, num_points);
  const auto spacetime_variables = kerr_schild.variables(
      coords, 0.,
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, Dim>,
                 gr::Tags::InverseSpatialMetric<DataVector, Dim>>{});
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(spacetime_variables);
  const auto& shift =
      get<gr::Tags::Shift<DataVector, Dim>>(spacetime_variables);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, Dim>>(spacetime_variables);

  auto normal_covector =
      make_with_random_values<tnsr::i<DataVector, Dim>>(gen, dist, num_points);
  const auto covector_mag = magnitude(normal_covector, inverse_spatial_metric);
  for (size_t i = 0; i < Dim; ++i) {
    normal_covector.get(i) /= covector_mag.get();
  }
  const auto normal_vector = tenex::evaluate<ti::I>(
      inverse_spatial_metric(ti::I, ti::J) * normal_covector(ti::j));
  CHECK_ITERABLE_APPROX(dot_product(normal_covector, normal_vector).get(),
                        DataVector(num_points, 1.));

  const auto psi_interior =
      make_with_random_values<Scalar<DataVector>>(gen, dist, num_points);
  const auto pi_interior =
      make_with_random_values<Scalar<DataVector>>(gen, dist, num_points);
  const auto phi_interior =
      make_with_random_values<tnsr::i<DataVector, 3>>(gen, dist, num_points);
  const auto gamma1 =
      make_with_random_values<Scalar<DataVector>>(gen, dist, num_points);
  const auto gamma2 =
      make_with_random_values<Scalar<DataVector>>(gen, dist, num_points);
  const auto dt_psi =
      make_with_random_values<Scalar<DataVector>>(gen, dist, num_points);
  const auto d_psi =
      make_with_random_values<tnsr::i<DataVector, 3>>(gen, dist, num_points);
  const auto d_phi =
      make_with_random_values<tnsr::ij<DataVector, 3>>(gen, dist, num_points);
  const auto worldtube_vars = make_with_random_values<Variables<
      tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                 CurvedScalarWave::Tags::Phi<Dim>>>>(gen, dist, num_points);

  Scalar<DataVector> psi_res(num_points);
  Scalar<DataVector> pi_res(num_points);
  tnsr::i<DataVector, Dim, Frame::Inertial> phi_res(num_points);
  Scalar<DataVector> lapse_res(num_points);
  tnsr::I<DataVector, Dim, Frame::Inertial> shift_res(num_points);
  Scalar<DataVector> gamma1_res(num_points);
  Scalar<DataVector> gamma2_res(num_points);
  tnsr::II<DataVector, Dim, Frame::Inertial> inverse_spatial_metric_res(
      num_points);

  CurvedScalarWave::BoundaryConditions::Worldtube<Dim> worldtube_bcs{};
  worldtube_bcs.dg_ghost(
      make_not_null(&psi_res), make_not_null(&pi_res), make_not_null(&phi_res),
      make_not_null(&lapse_res), make_not_null(&shift_res),
      make_not_null(&gamma1_res), make_not_null(&gamma2_res),
      make_not_null(&inverse_spatial_metric_res), std::nullopt, normal_covector,
      normal_vector, psi_interior, pi_interior, phi_interior, lapse, shift,
      inverse_spatial_metric, gamma1, gamma2, dt_psi, d_psi, d_phi,
      worldtube_vars);

  CHECK(lapse_res == lapse);
  CHECK(shift_res == shift);
  CHECK(inverse_spatial_metric_res == inverse_spatial_metric);
  CHECK(gamma1_res == gamma1);
  CHECK(gamma2_res == gamma2);
  const auto unit_normal_vector = tenex::evaluate<ti::I>(
      inverse_spatial_metric(ti::I, ti::J) * normal_covector(ti::j));
  const auto char_fields_interior = CurvedScalarWave::characteristic_fields(
      gamma2, psi_interior, pi_interior, phi_interior, normal_covector,
      normal_vector);

  const auto char_fields_worldtube = CurvedScalarWave::characteristic_fields(
      gamma2, get<CurvedScalarWave::Tags::Psi>(worldtube_vars),
      get<CurvedScalarWave::Tags::Pi>(worldtube_vars),
      get<CurvedScalarWave::Tags::Phi<Dim>>(worldtube_vars), normal_covector,
      normal_vector);

  const auto char_fields_res = CurvedScalarWave::characteristic_fields(
      gamma2, psi_res, pi_res, phi_res, normal_covector, normal_vector);
  Approx approx = Approx::custom().epsilon(1.e-12).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get<CurvedScalarWave::Tags::VPsi>(char_fields_interior),
      get<CurvedScalarWave::Tags::VPsi>(char_fields_res), approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get<CurvedScalarWave::Tags::VZero<Dim>>(char_fields_interior),
      get<CurvedScalarWave::Tags::VZero<Dim>>(char_fields_res), approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get<CurvedScalarWave::Tags::VPlus>(char_fields_interior),
      get<CurvedScalarWave::Tags::VPlus>(char_fields_res), approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get<CurvedScalarWave::Tags::VMinus>(char_fields_worldtube),
      get<CurvedScalarWave::Tags::VMinus>(char_fields_res), approx);
}

// this takes a variables on the C++ side filled with 1 and returns only
// 1 on the python side, point by point. There is little point in trying to
// transfer the variables to the python side because it is unclear how the
// different components should be packaged and ordered.
template <size_t NumGridPoints>
struct ConvertWorldtubeSolution {
  using unpacked_container = double;
  using packed_container = Variables<
      tmpl::list<CurvedScalarWave::Tags::Psi, ::CurvedScalarWave::Tags::Pi,
                 CurvedScalarWave::Tags::Phi<3>>>;
  using packed_type = packed_container;
  static constexpr size_t face_size = NumGridPoints * NumGridPoints;

  static packed_container create_container() {
    return packed_container{face_size, 1.};
  }

  static inline unpacked_container unpack(const packed_container /*packed*/,
                                          const size_t /*grid_point_index*/) {
    return 1.;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    *packed = packed_container{face_size, 1.};
  }

  static inline size_t get_size(const packed_container& packed) {
    return packed.number_of_grid_points();
  }
};

void test_python(const gsl::not_null<std::mt19937*> gen) {
  static constexpr size_t Dim = 3;
  std::uniform_real_distribution<> dist(-10., 10.);
  static constexpr size_t num_grid_points = 5;
  auto box = db::create<db::AddSimpleTags<
      CurvedScalarWave::Worldtube::Tags::WorldtubeSolution<Dim>>>(
      ConvertWorldtubeSolution<num_grid_points>::create_container());

  namespace helpers = TestHelpers::evolution::dg;
  helpers::test_boundary_condition_with_python<
      CurvedScalarWave::BoundaryConditions::Worldtube<Dim>,
      CurvedScalarWave::BoundaryConditions::BoundaryCondition<Dim>,
      CurvedScalarWave::System<Dim>,
      tmpl::list<CurvedScalarWave::BoundaryCorrections::UpwindPenalty<Dim>>,
      tmpl::list<ConvertWorldtubeSolution<num_grid_points>>,
      tmpl::list<CurvedScalarWave::Worldtube::Tags::WorldtubeSolution<Dim>>>(
      gen,
      "Evolution.Systems.CurvedScalarWave.BoundaryConditions."
      "Worldtube",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<
              ::Tags::dt<CurvedScalarWave::Tags::Psi>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::dt<CurvedScalarWave::Tags::Pi>>,
          helpers::Tags::PythonFunctionName<
              ::Tags::dt<CurvedScalarWave::Tags::Phi<Dim>>>,
          helpers::Tags::PythonFunctionName<CurvedScalarWave::Tags::Psi>,
          helpers::Tags::PythonFunctionName<CurvedScalarWave::Tags::Pi>,
          helpers::Tags::PythonFunctionName<CurvedScalarWave::Tags::Phi<Dim>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Lapse<DataVector>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Shift<DataVector, Dim>>,
          helpers::Tags::PythonFunctionName<
              gr::Tags::InverseSpatialMetric<DataVector, Dim>>,
          helpers::Tags::PythonFunctionName<
              CurvedScalarWave::Tags::ConstraintGamma1>,
          helpers::Tags::PythonFunctionName<
              CurvedScalarWave::Tags::ConstraintGamma2>>{
          "error", "dt_psi_worldtube", "dt_pi_worldtube", "dt_phi_worldtube",
          "psi", "pi", "phi", "lapse", "shift", "inverse_spatial_metric",
          "gamma1", "gamma2"},
      "Worldtube:\n", Index<Dim - 1>{num_grid_points}, box,
      tuples::TaggedTuple<>{});
}

}  // namespace
SPECTRE_TEST_CASE("Unit.Evolution.Systems.CSW.BoundaryConditions.Worldtube",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  MAKE_GENERATOR(gen);
  test_dg_ghost(make_not_null(&gen));
  test_python(make_not_null(&gen));
}
