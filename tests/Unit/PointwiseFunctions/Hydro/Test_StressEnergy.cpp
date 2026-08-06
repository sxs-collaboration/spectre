// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/StressEnergy.hpp"

namespace {
// Verifies that the trace-reversed stress-energy tensor, computed from the
// upper-index stress-energy tensor, matches the result of the function
// "trace_reversed_stress_energy" from
// Evolution/Systems/GrMhd/GhValenciaDivClean/StressEnergy.cpp.
// Note: The function "trace_reversed_stress_energy" only supports DataVector,
// so this test is restricted to DataVector.
void consistency_check(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);

  // Generate random lapse, shift, spatial metric.
  const auto lapse =
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size);
  const auto shift = TestHelpers::gr::random_shift<3>(make_not_null(&generator),
                                                      used_for_size);
  const auto spatial_metric = TestHelpers::gr::random_spatial_metric<3>(
      make_not_null(&generator), used_for_size);
  const tnsr::II<DataVector, 3, Frame::Inertial> inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  std::uniform_real_distribution<> distribution1(-1.0, 1.0);
  std::uniform_real_distribution<> distribution2(0.01, 1.0);

  // Generate random hydro variables.
  auto spatial_velocity = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&distribution1), used_for_size);
  const auto magnetic_field = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&distribution1), used_for_size);

  const auto rest_mass_density = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution2), used_for_size);
  const auto pressure = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution2), used_for_size);
  const auto specific_internal_energy =
      make_with_random_values<Scalar<DataVector>>(make_not_null(&generator),
                                                  make_not_null(&distribution2),
                                                  used_for_size);

  TempBuffer<tmpl::list<
      ::Tags::TempScalar<0, DataVector>, ::Tags::TempScalar<1, DataVector>,
      ::Tags::TempScalar<2, DataVector>, ::Tags::TempScalar<3, DataVector>,
      ::Tags::TempScalar<4, DataVector>, ::Tags::TempScalar<5, DataVector>,
      ::Tags::Tempi<6, 3, Frame::Inertial, DataVector>,
      ::Tags::Tempi<7, 3, Frame::Inertial, DataVector>,
      ::Tags::Tempaa<8, 3, Frame::Inertial, DataVector>,
      ::Tags::Tempaa<9, 3, Frame::Inertial, DataVector>,
      ::Tags::Tempaa<10, 3, Frame::Inertial, DataVector>,
      ::Tags::Tempaa<11, 3, Frame::Inertial, DataVector>,
      ::Tags::TempAA<12, 3, Frame::Inertial, DataVector>,
      ::Tags::TempAA<13, 3, Frame::Inertial, DataVector>>>
      buffer(get_size(used_for_size));

  auto& spatial_velocity_squared =
      get<::Tags::TempScalar<0, DataVector>>(buffer);
  auto& magnetic_field_squared = get<::Tags::TempScalar<1, DataVector>>(buffer);
  auto& magnetic_field_dot_spatial_velocity =
      get<::Tags::TempScalar<2, DataVector>>(buffer);
  auto& lorentz_factor_v = get<::Tags::TempScalar<3, DataVector>>(buffer);
  auto& one_over_w_squared = get<::Tags::TempScalar<4, DataVector>>(buffer);
  auto& comoving_magnetic_field_magnitude_v =
      get<::Tags::TempScalar<5, DataVector>>(buffer);

  auto& spatial_velocity_one_form =
      get<::Tags::Tempi<6, 3, Frame::Inertial, DataVector>>(buffer);
  auto& magnetic_field_one_form =
      get<::Tags::Tempi<7, 3, Frame::Inertial, DataVector>>(buffer);

  auto& spacetime_metric_v =
      get<::Tags::Tempaa<8, 3, Frame::Inertial, DataVector>>(buffer);
  auto& stress_energy_tensor_lowered =
      get<::Tags::Tempaa<9, 3, Frame::Inertial, DataVector>>(buffer);
  auto& trace_reversed_stress_energy_tensor_v =
      get<::Tags::Tempaa<10, 3, Frame::Inertial, DataVector>>(buffer);
  auto& trace_reversed_stress_energy_tensor_calc =
      get<::Tags::Tempaa<11, 3, Frame::Inertial, DataVector>>(buffer);
  auto& inverse_spacetime_metric_v =
      get<::Tags::TempAA<12, 3, Frame::Inertial, DataVector>>(buffer);
  auto& stress_energy_tensor_v =
      get<::Tags::TempAA<13, 3, Frame::Inertial, DataVector>>(buffer);

  dot_product(make_not_null(&spatial_velocity_squared), spatial_velocity,
              spatial_velocity, spatial_metric);

  // To avoid speeds exceeding the speed of light,
  // normalize the spatial velocity so that its magnitude is 0.6c.
  for (size_t i = 0; i < 3; ++i) {
    spatial_velocity.get(i) /= (5. / 3.) * sqrt(get(spatial_velocity_squared));
  }
  get(spatial_velocity_squared) = 9. / 25.;

  dot_product(make_not_null(&magnetic_field_squared), magnetic_field,
              magnetic_field, spatial_metric);
  dot_product(make_not_null(&magnetic_field_dot_spatial_velocity),
              magnetic_field, spatial_velocity, spatial_metric);
  hydro::lorentz_factor(make_not_null(&lorentz_factor_v),
                        spatial_velocity_squared);
  get(one_over_w_squared) = 1. / square(get(lorentz_factor_v));

  tenex::evaluate<ti::i>(
      make_not_null(&spatial_velocity_one_form),
      spatial_velocity(ti::J) * spatial_metric(ti::j, ti::i));

  tenex::evaluate<ti::i>(make_not_null(&magnetic_field_one_form),
                         magnetic_field(ti::J) * spatial_metric(ti::j, ti::i));

  hydro::comoving_magnetic_field_squared(
      make_not_null(&comoving_magnetic_field_magnitude_v),
      magnetic_field_squared, magnetic_field_dot_spatial_velocity,
      lorentz_factor_v);
  get(comoving_magnetic_field_magnitude_v) =
      sqrt(get(comoving_magnetic_field_magnitude_v));

  hydro::stress_energy_tensor(
      make_not_null(&stress_energy_tensor_v), rest_mass_density,
      specific_internal_energy, pressure, lorentz_factor_v, lapse,
      comoving_magnetic_field_magnitude_v, spatial_velocity, shift,
      magnetic_field, spatial_metric, inverse_spatial_metric);

  gr::spacetime_metric(make_not_null(&spacetime_metric_v), lapse, shift,
                       spatial_metric);

  gr::inverse_spacetime_metric(make_not_null(&inverse_spacetime_metric_v),
                               lapse, shift, inverse_spatial_metric);

  tnsr::a<DataVector, 3> four_velocity_one_form_buffer;
  tnsr::a<DataVector, 3> comoving_magnetic_field_one_form_buffer;
  ::grmhd::GhValenciaDivClean::trace_reversed_stress_energy(
      make_not_null(&trace_reversed_stress_energy_tensor_v),
      make_not_null(&four_velocity_one_form_buffer),
      make_not_null(&comoving_magnetic_field_one_form_buffer),
      rest_mass_density, spatial_velocity_one_form, magnetic_field_one_form,
      magnetic_field_squared, magnetic_field_dot_spatial_velocity,
      lorentz_factor_v, one_over_w_squared, pressure, specific_internal_energy,
      spacetime_metric_v, shift, lapse);

  tenex::evaluate<ti::a, ti::b>(make_not_null(&stress_energy_tensor_lowered),
                                stress_energy_tensor_v(ti::C, ti::D) *
                                    spacetime_metric_v(ti::c, ti::a) *
                                    spacetime_metric_v(ti::d, ti::b));
  tenex::evaluate<ti::a, ti::b>(
      make_not_null(&trace_reversed_stress_energy_tensor_calc),
      stress_energy_tensor_lowered(ti::a, ti::b) -
          0.5 * spacetime_metric_v(ti::a, ti::b) *
              inverse_spacetime_metric_v(ti::C, ti::D) *
              stress_energy_tensor_lowered(ti::d, ti::c));

  // Test cases sometimes fail with the default scale/epsilon value due to
  // catastrophic cancellation in the computation for the test.
  // Therefore, we use a custom scale and epsilon values here.
  Approx approx = Approx::custom().epsilon(1.e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(trace_reversed_stress_energy_tensor_calc,
                               trace_reversed_stress_energy_tensor_v, approx);
}
}  // namespace

namespace hydro {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.StressEnergy",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/Python/");
  const DataVector used_for_size(5);
  const double tolerance(1.e-11);
  pypp::check_with_random_values<1>(&energy_density<DataVector>,
                                    "Test_StressEnergy", {"energy_density"},
                                    {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(&momentum_density<DataVector>,
                                    "Test_StressEnergy", {"momentum_density"},
                                    {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(&stress_trace<DataVector>,
                                    "Test_StressEnergy", {"stress_trace"},
                                    {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &stress_energy_tensor<DataVector>, "Test_StressEnergy",
      {"stress_energy_tensor"}, {{{0.0, 1.0}}}, used_for_size, tolerance);

  consistency_check(used_for_size);
  TestHelpers::db::test_compute_tag<
      hydro::Tags::StressEnergyCompute<DataVector>>("StressEnergy");
}

}  // namespace hydro
