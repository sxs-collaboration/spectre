// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {
template <size_t Index, size_t SpatialDim>
Scalar<DataVector> speed_with_index(
    const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, SpatialDim, Frame::Inertial>& shift,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& normal) {
  return Scalar<DataVector>{CurvedScalarWave::characteristic_speeds(
      gamma_1, lapse, shift, normal)[Index]};
}

template <size_t SpatialDim>
void test_characteristic_speeds() noexcept {
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(speed_with_index<0, SpatialDim>,
                                    "Characteristics", "char_speed_vpsi",
                                    {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<1, SpatialDim>,
                                    "Characteristics", "char_speed_vzero",
                                    {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<3, SpatialDim>,
                                    "Characteristics", "char_speed_vminus",
                                    {{{-10.0, 10.0}}}, used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<2, SpatialDim>,
                                    "Characteristics", "char_speed_vplus",
                                    {{{-10.0, 10.0}}}, used_for_size);
}
}  // namespace

namespace {
template <typename Tag, size_t SpatialDim>
typename Tag::type field_with_tag(
    const Scalar<DataVector>& gamma_2,
    const tnsr::II<DataVector, SpatialDim, Frame::Inertial>&
        inverse_spatial_metric,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& normal_one_form) {
  return get<Tag>(CurvedScalarWave::characteristic_fields(
      gamma_2, inverse_spatial_metric, psi, pi, phi, normal_one_form));
}

template <size_t SpatialDim>
void test_characteristic_fields() noexcept {
  const DataVector used_for_size(5);
  // VPsi
  pypp::check_with_random_values<1>(
      field_with_tag<CurvedScalarWave::Tags::VPsi, SpatialDim>,
      "Characteristics", "char_field_vpsi", {{{-10., 10.}}}, used_for_size);
  // VZero
  pypp::check_with_random_values<1>(
      field_with_tag<CurvedScalarWave::Tags::VZero<SpatialDim>, SpatialDim>,
      "Characteristics", "char_field_vzero", {{{-10., 10.}}}, used_for_size);
  // VPlus
  pypp::check_with_random_values<1>(
      field_with_tag<CurvedScalarWave::Tags::VPlus, SpatialDim>,
      "Characteristics", "char_field_vplus", {{{-10., 10.}}}, used_for_size);
  // VMinus
  pypp::check_with_random_values<1>(
      field_with_tag<CurvedScalarWave::Tags::VMinus, SpatialDim>,
      "Characteristics", "char_field_vminus", {{{-10., 10.}}}, used_for_size);
}
}  // namespace

namespace {
template <typename Tag, size_t SpatialDim>
typename Tag::type evolved_field_with_tag(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& u_psi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& u_zero,
    const Scalar<DataVector>& u_plus, const Scalar<DataVector>& u_minus,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& normal_one_form) {
  return get<Tag>(CurvedScalarWave::evolved_fields_from_characteristic_fields(
      gamma_2, u_psi, u_zero, u_plus, u_minus, normal_one_form));
}

template <size_t SpatialDim>
void test_evolved_from_characteristic_fields() noexcept {
  const DataVector used_for_size(5);
  // Psi
  pypp::check_with_random_values<1>(
      evolved_field_with_tag<CurvedScalarWave::Psi, SpatialDim>,
      "Characteristics", "evol_field_psi", {{{-10., 10.}}}, used_for_size);
  // Pi
  pypp::check_with_random_values<1>(
      evolved_field_with_tag<CurvedScalarWave::Pi, SpatialDim>,
      "Characteristics", "evol_field_pi", {{{-10., 10.}}}, used_for_size);
  // Phi
  pypp::check_with_random_values<1>(
      evolved_field_with_tag<CurvedScalarWave::Phi<SpatialDim>, SpatialDim>,
      "Characteristics", "evol_field_phi", {{{-10., 10.}}}, used_for_size);
}

template <size_t SpatialDim>
void test_characteristics_compute_tags() noexcept {
  TestHelpers::db::test_compute_tag<
      CurvedScalarWave::CharacteristicSpeedsCompute<SpatialDim>>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_compute_tag<
      CurvedScalarWave::CharacteristicFieldsCompute<SpatialDim>>(
      "CharacteristicFields");
  TestHelpers::db::test_compute_tag<
      CurvedScalarWave::EvolvedFieldsFromCharacteristicFieldsCompute<
          SpatialDim>>("EvolvedFieldsFromCharacteristicFields");

  const DataVector used_for_size(5);

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.0, 1.0);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  // Randomized tensors
  const auto gamma_1 = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto gamma_2 = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto shift =
      make_with_random_values<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);
  const auto inverse_spatial_metric = make_with_random_values<
      tnsr::II<DataVector, SpatialDim, Frame::Inertial>>(
      nn_generator, nn_distribution, used_for_size);
  const auto psi = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto pi = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto phi =
      make_with_random_values<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);
  const auto normal =
      make_with_random_values<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);

  // Insert into databox
  const auto box = db::create<
      db::AddSimpleTags<
          CurvedScalarWave::Tags::ConstraintGamma1,
          CurvedScalarWave::Tags::ConstraintGamma2, gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<SpatialDim, Frame::Inertial, DataVector>,
          gr::Tags::InverseSpatialMetric<SpatialDim, Frame::Inertial,
                                         DataVector>,
          CurvedScalarWave::Psi, CurvedScalarWave::Pi,
          CurvedScalarWave::Phi<SpatialDim>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<SpatialDim>>>,
      db::AddComputeTags<
          CurvedScalarWave::CharacteristicSpeedsCompute<SpatialDim>,
          CurvedScalarWave::CharacteristicFieldsCompute<SpatialDim>>>(
      gamma_1, gamma_2, lapse, shift, inverse_spatial_metric, psi, pi, phi,
      normal);
  // Test compute tag for char speeds
  CHECK(
      db::get<CurvedScalarWave::Tags::CharacteristicSpeeds<SpatialDim>>(box) ==
      CurvedScalarWave::characteristic_speeds(gamma_1, lapse, shift, normal));
  // Test compute tag for char fields
  CHECK(
      db::get<CurvedScalarWave::Tags::CharacteristicFields<SpatialDim>>(box) ==
      CurvedScalarWave::characteristic_fields(gamma_2, inverse_spatial_metric,
                                              psi, pi, phi, normal));

  // more randomized tensors
  const auto u_psi = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto u_zero =
      make_with_random_values<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          nn_generator, nn_distribution, used_for_size);
  const auto u_plus = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto u_minus = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  // Insert into databox
  const auto box2 = db::create<
      db::AddSimpleTags<
          CurvedScalarWave::Tags::ConstraintGamma2,
          CurvedScalarWave::Tags::VPsi,
          CurvedScalarWave::Tags::VZero<SpatialDim>,
          CurvedScalarWave::Tags::VPlus, CurvedScalarWave::Tags::VMinus,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<SpatialDim>>>,
      db::AddComputeTags<
          CurvedScalarWave::EvolvedFieldsFromCharacteristicFieldsCompute<
              SpatialDim>>>(gamma_2, u_psi, u_zero, u_plus, u_minus, normal);
  // Test compute tag for evolved fields computed from char fields
  CHECK(db::get<CurvedScalarWave::Tags::EvolvedFieldsFromCharacteristicFields<
            SpatialDim>>(box2) ==
        CurvedScalarWave::evolved_fields_from_characteristic_fields(
            gamma_2, u_psi, u_zero, u_plus, u_minus, normal));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/CurvedScalarWave/"};

  test_characteristic_speeds<1>();
  test_characteristic_speeds<2>();
  test_characteristic_speeds<3>();

  test_characteristic_fields<1>();
  test_characteristic_fields<2>();
  test_characteristic_fields<3>();

  test_evolved_from_characteristic_fields<1>();
  test_evolved_from_characteristic_fields<2>();
  test_evolved_from_characteristic_fields<3>();

  test_characteristics_compute_tags<1>();
  test_characteristics_compute_tags<2>();
  test_characteristics_compute_tags<3>();

  CHECK(CurvedScalarWave::ComputeLargestCharacteristicSpeed<1>::apply(
            {{DataVector{1., -4., 3.4, 2., 5.},
              DataVector{22., -8., 190., 6., 4.},
              DataVector{1., -7., 31., 2., 5.},
              DataVector{7., 8.9, 4., 2., 1.}}}) == 190.);
  CHECK(CurvedScalarWave::ComputeLargestCharacteristicSpeed<1>::apply(
            {{DataVector{1., 4., 3., 2., 5.}, DataVector{2., 8., -10., 6., 4.},
              DataVector{1., 7., 3., -11., 5.},
              DataVector{7., 3., 4., 2., 1.}}}) == 11.);

  CHECK(CurvedScalarWave::ComputeLargestCharacteristicSpeed<2>::apply(
            {{DataVector{1., -4., 3.4, 2., 5.},
              DataVector{22., -8., 190., 6., 4.},
              DataVector{1., -7., 31., 2., 5.},
              DataVector{7., 8.9, 4., 2., 1.}}}) == 190.);
  CHECK(CurvedScalarWave::ComputeLargestCharacteristicSpeed<2>::apply(
            {{DataVector{1., 4., 3., 2., 5.}, DataVector{2., 8., -10., 6., 4.},
              DataVector{1., 7., 3., -11., 5.},
              DataVector{7., 3., 4., 2., 1.}}}) == 11.);

  CHECK(CurvedScalarWave::ComputeLargestCharacteristicSpeed<3>::apply(
            {{DataVector{1., -4., 3.4, 2., 5.},
              DataVector{22., -8., 190., 6., 4.},
              DataVector{1., -7., 31., 2., 5.},
              DataVector{7., 8.9, 4., 2., 1.}}}) == 190.);
  CHECK(CurvedScalarWave::ComputeLargestCharacteristicSpeed<3>::apply(
            {{DataVector{1., 4., 3., 2., 5.}, DataVector{2., 8., -10., 6., 4.},
              DataVector{1., 7., 3., -11., 5.},
              DataVector{7., 3., 4., 2., 1.}}}) == 11.);
}
