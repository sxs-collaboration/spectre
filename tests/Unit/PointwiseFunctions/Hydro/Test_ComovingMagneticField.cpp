// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "PointwiseFunctions/Hydro/ComovingMagneticField.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// Verifies that the squared comoving magnetic field, computed as the dot
// product of the comoving magnetic field and its one-form, matches the value
// returned by the function "comoving_magnetic_field_squared".
template <typename DataType>
void consistency_check(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  const auto lapse =
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size);
  const auto shift = TestHelpers::gr::random_shift<3>(make_not_null(&generator),
                                                      used_for_size);
  const auto spatial_metric = TestHelpers::gr::random_spatial_metric<3>(
      make_not_null(&generator), used_for_size);

  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  auto spatial_velocity = make_with_random_values<tnsr::I<DataType, 3>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  const auto magnetic_field = make_with_random_values<tnsr::I<DataType, 3>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  TempBuffer<tmpl::list<
      ::Tags::TempScalar<0, DataType>, ::Tags::TempScalar<1, DataType>,
      ::Tags::TempScalar<2, DataType>, ::Tags::TempScalar<3, DataType>,
      ::Tags::Tempi<4, 3, Frame::Inertial, DataType>,
      ::Tags::Tempi<5, 3, Frame::Inertial, DataType>>>
      buffer(get_size(used_for_size));

  auto& spatial_velocity_squared = get<::Tags::TempScalar<0, DataType>>(buffer);
  auto& magnetic_field_squared = get<::Tags::TempScalar<1, DataType>>(buffer);
  auto& magnetic_field_dot_spatial_velocity =
      get<::Tags::TempScalar<2, DataType>>(buffer);
  auto& lorentz_factor_v = get<::Tags::TempScalar<3, DataType>>(buffer);
  auto& spatial_velocity_one_form =
      get<::Tags::Tempi<4, 3, Frame::Inertial, DataType>>(buffer);
  auto& magnetic_field_one_form =
      get<::Tags::Tempi<5, 3, Frame::Inertial, DataType>>(buffer);

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

  tenex::evaluate<ti::i>(
      make_not_null(&spatial_velocity_one_form),
      spatial_metric(ti::i, ti::j) * spatial_velocity(ti::J));

  tenex::evaluate<ti::i>(make_not_null(&magnetic_field_one_form),
                         spatial_metric(ti::i, ti::j) * magnetic_field(ti::J));

  const tnsr::A<DataType, 3> comoving_magnetic_field_v =
      hydro::comoving_magnetic_field(spatial_velocity, magnetic_field,
                                     magnetic_field_dot_spatial_velocity,
                                     lorentz_factor_v, shift, lapse);

  const tnsr::a<DataType, 3> comoving_magnetic_field_one_form_v =
      hydro::comoving_magnetic_field_one_form(
          spatial_velocity_one_form, magnetic_field_one_form,
          magnetic_field_dot_spatial_velocity, lorentz_factor_v, shift, lapse);

  const Scalar<DataType> comoving_magnetic_field_squared_v =
      hydro::comoving_magnetic_field_squared(
          magnetic_field_squared, magnetic_field_dot_spatial_velocity,
          lorentz_factor_v);

  auto comoving_magnetic_field_squared_calc =
      make_with_value<Scalar<DataType>>(used_for_size, 0.0);
  for (size_t i = 0; i < 4; ++i) {
    get(comoving_magnetic_field_squared_calc) +=
        comoving_magnetic_field_v.get(i) *
        comoving_magnetic_field_one_form_v.get(i);
  }
  CHECK_ITERABLE_APPROX(comoving_magnetic_field_squared_calc,
                        comoving_magnetic_field_squared_v);
}

#ifdef SPECTRE_KOKKOS
// Evaluates all functions for `double`s, so they can be compared between a
// Kokkos kernel and the host
constexpr size_t number_of_values = 18;
KOKKOS_FUNCTION std::array<double, number_of_values> evaluate_all(
    const tnsr::I<double, 3>& spatial_velocity,
    const tnsr::I<double, 3>& magnetic_field,
    const tnsr::i<double, 3>& spatial_velocity_one_form,
    const tnsr::i<double, 3>& magnetic_field_one_form,
    const Scalar<double>& magnetic_field_squared,
    const Scalar<double>& magnetic_field_dot_spatial_velocity,
    const Scalar<double>& lorentz_factor, const tnsr::I<double, 3>& shift,
    const Scalar<double>& lapse) {
  tnsr::A<double, 3> vector{};
  hydro::comoving_magnetic_field(
      make_not_null(&vector), spatial_velocity, magnetic_field,
      magnetic_field_dot_spatial_velocity, lorentz_factor, shift, lapse);
  const auto returned_vector = hydro::comoving_magnetic_field(
      spatial_velocity, magnetic_field, magnetic_field_dot_spatial_velocity,
      lorentz_factor, shift, lapse);
  tnsr::a<double, 3> one_form{};
  hydro::comoving_magnetic_field_one_form(
      make_not_null(&one_form), spatial_velocity_one_form,
      magnetic_field_one_form, magnetic_field_dot_spatial_velocity,
      lorentz_factor, shift, lapse);
  const auto returned_one_form = hydro::comoving_magnetic_field_one_form(
      spatial_velocity_one_form, magnetic_field_one_form,
      magnetic_field_dot_spatial_velocity, lorentz_factor, shift, lapse);
  Scalar<double> squared{};
  hydro::comoving_magnetic_field_squared(
      make_not_null(&squared), magnetic_field_squared,
      magnetic_field_dot_spatial_velocity, lorentz_factor);
  const auto returned_squared = hydro::comoving_magnetic_field_squared(
      magnetic_field_squared, magnetic_field_dot_spatial_velocity,
      lorentz_factor);
  return {{get<0>(vector), get<1>(vector), get<2>(vector), get<3>(vector),
           get<0>(returned_vector), get<1>(returned_vector),
           get<2>(returned_vector), get<3>(returned_vector), get<0>(one_form),
           get<1>(one_form), get<2>(one_form), get<3>(one_form),
           get<0>(returned_one_form), get<1>(returned_one_form),
           get<2>(returned_one_form), get<3>(returned_one_form), get(squared),
           get(returned_squared)}};
}

// The functions give the same results in a Kokkos kernel as on the host
void test_on_device() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 1.0);
  const auto spatial_velocity = make_with_random_values<tnsr::I<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto magnetic_field = make_with_random_values<tnsr::I<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto spatial_velocity_one_form =
      make_with_random_values<tnsr::i<double, 3>>(
          make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto magnetic_field_one_form =
      make_with_random_values<tnsr::i<double, 3>>(
          make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto magnetic_field_squared = make_with_random_values<Scalar<double>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto magnetic_field_dot_spatial_velocity =
      make_with_random_values<Scalar<double>>(
          make_not_null(&generator), make_not_null(&distribution), 0.0);
  const Scalar<double> lorentz_factor{1.5};
  const auto shift = make_with_random_values<tnsr::I<double, 3>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);
  const auto lapse = make_with_random_values<Scalar<double>>(
      make_not_null(&generator), make_not_null(&distribution), 0.0);

  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    return evaluate_all(
        spatial_velocity, magnetic_field, spatial_velocity_one_form,
        magnetic_field_one_form, magnetic_field_squared,
        magnetic_field_dot_spatial_velocity, lorentz_factor, shift, lapse);
  });
}
#endif  // SPECTRE_KOKKOS
}  // namespace

namespace hydro {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.ComovingMagneticField",
                  "[Unit][Hydro]") {
  const pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<DataVector, 3> (*)(
          const tnsr::I<DataVector, 3>&, const tnsr::I<DataVector, 3>&,
          const Scalar<DataVector>&, const Scalar<DataVector>&,
          const tnsr::I<DataVector, 3>&, const Scalar<DataVector>&)>(
          &comoving_magnetic_field<DataVector>),
      "ComovingMagneticField", "comoving_magnetic_field", {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataVector, 3> (*)(
          const tnsr::i<DataVector, 3>&, const tnsr::i<DataVector, 3>&,
          const Scalar<DataVector>&, const Scalar<DataVector>&,
          const tnsr::I<DataVector, 3>&, const Scalar<DataVector>&)>(
          &comoving_magnetic_field_one_form<DataVector>),
      "ComovingMagneticField", "comoving_magnetic_field_one_form",
      {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataVector> (*)(const Scalar<DataVector>&,
                                         const Scalar<DataVector>&,
                                         const Scalar<DataVector>&)>(
          &comoving_magnetic_field_squared<DataVector>),
      "ComovingMagneticField", "comoving_magnetic_field_squared",
      {{{0.0, 1.0}}}, used_for_size);

  consistency_check(used_for_size);
#ifdef SPECTRE_KOKKOS
  test_on_device();
#endif  // SPECTRE_KOKKOS
}

}  // namespace hydro
