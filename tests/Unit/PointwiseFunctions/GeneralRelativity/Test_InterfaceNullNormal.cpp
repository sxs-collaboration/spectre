// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeRandomVectorInMagnitudeRange.hpp"
#include "Helpers/DataStructures/RandomUnitNormal.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/InterfaceNullNormal.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);

template <size_t SpatialDim, typename DataType>
tnsr::a<DataType, SpatialDim, Frame::Inertial>
interface_outgoing_null_normal_one_form(
    const tnsr::a<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_one_form,
    const tnsr::i<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_one_form,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>& shift) {
  return gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
      spacetime_normal_one_form, interface_normal_one_form, shift, 1.);
}
template <size_t SpatialDim, typename DataType>
tnsr::a<DataType, SpatialDim, Frame::Inertial>
interface_incoming_null_normal_one_form(
    const tnsr::a<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_one_form,
    const tnsr::i<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_one_form,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>& shift) {
  return gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
      spacetime_normal_one_form, interface_normal_one_form, shift, -1.);
}
template <size_t SpatialDim, typename DataType>
tnsr::A<DataType, SpatialDim, Frame::Inertial>
interface_outgoing_null_normal_vector(
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_vector,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_vector) {
  return gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
      spacetime_normal_vector, interface_normal_vector, 1.);
}
template <size_t SpatialDim, typename DataType>
tnsr::A<DataType, SpatialDim, Frame::Inertial>
interface_incoming_null_normal_vector(
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_vector,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>&
        interface_normal_vector) {
  return gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
      spacetime_normal_vector, interface_normal_vector, -1.);
}

template <size_t SpatialDim, typename DataType>
tnsr::A<DataType, SpatialDim, Frame::Inertial>
extend_spatial_vector_to_spacetime(
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>& spatial_vector,
    const DataType& used_for_size) {
  auto spacetime_vector =
      make_with_value<tnsr::A<DataType, SpatialDim, Frame::Inertial>>(
          used_for_size, 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    spacetime_vector.get(i + 1) = spatial_vector.get(i);
  }
  return spacetime_vector;
}

template <size_t SpatialDim, typename DataType>
tnsr::a<DataType, SpatialDim, Frame::Inertial>
extend_spatial_one_form_to_spacetime(
    const tnsr::i<DataType, SpatialDim, Frame::Inertial>& spatial_one_form,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>& shift,
    const DataType& used_for_size) {
  auto spacetime_one_form =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame::Inertial>>(
          used_for_size, 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    spacetime_one_form.get(i + 1) = spatial_one_form.get(i);
  }
  spacetime_one_form.get(0) = get(dot_product(shift, spatial_one_form));
  return spacetime_one_form;
}

template <size_t SpatialDim, typename DataType>
void test_interface_null_normals(const DataType& used_for_size) {
  {
    auto* f = &interface_outgoing_null_normal_one_form<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_outgoing_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
  {
    auto* f = &interface_outgoing_null_normal_vector<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_outgoing_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
  {
    auto* f = &interface_incoming_null_normal_one_form<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_incoming_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
  {
    auto* f = &interface_incoming_null_normal_vector<SpatialDim, DataType>;
    pypp::check_with_random_values<1>(f, "InterfaceNullNormal",
                                      "interface_incoming_null_normal",
                                      {{{-1., 1.}}}, used_for_size);
  }
}

template <typename DataType>
void test_one_form_time_component_difference(const DataType& used_for_size) {
  constexpr size_t spatial_dim = 3;
  MAKE_GENERATOR(generator);
  const auto lapse =
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size);
  const auto shift = TestHelpers::gr::random_shift<spatial_dim>(
      make_not_null(&generator), used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<spatial_dim>(
          make_not_null(&generator), used_for_size);
  const auto interface_normal_vector =
      random_unit_normal(make_not_null(&generator), spatial_metric);
  auto interface_normal_one_form =
      make_with_value<tnsr::i<DataType, spatial_dim, Frame::Inertial>>(
          used_for_size, 0.0);
  raise_or_lower_index(make_not_null(&interface_normal_one_form),
                       interface_normal_vector, spatial_metric);

  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<DataType, spatial_dim, Frame::Inertial>(
          lapse);

  auto incoming =
      gr::interface_null_normal<DataType, spatial_dim, Frame::Inertial>(
          spacetime_normal_one_form, interface_normal_one_form, shift, -1.0);
  auto outgoing =
      gr::interface_null_normal<DataType, spatial_dim, Frame::Inertial>(
          spacetime_normal_one_form, interface_normal_one_form, shift, 1.0);

  const auto shift_dot_interface_normal =
      dot_product(shift, interface_normal_one_form);
  const DataType expected_time_component_difference =
      std::numbers::sqrt2 * get(shift_dot_interface_normal);
  const DataType computed_time_component_difference =
      outgoing.get(0) - incoming.get(0);
  CHECK_ITERABLE_CUSTOM_APPROX(computed_time_component_difference,
                               expected_time_component_difference,
                               custom_approx);

  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);
  const auto incoming_outgoing_difference =
      tenex::evaluate<ti::a>(outgoing(ti::a) - incoming(ti::a));
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(spacetime_normal_vector, incoming_outgoing_difference)),
      make_with_value<DataType>(used_for_size, 0.0), custom_approx);
}

template <size_t SpatialDim, typename DataType>
void check_null_normal_contractions(
    const tnsr::aa<DataType, SpatialDim, Frame::Inertial>& spacetime_metric,
    const tnsr::AA<DataType, SpatialDim, Frame::Inertial>&
        inverse_spacetime_metric,
    const tnsr::a<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_vector,
    const tnsr::i<DataType, SpatialDim, Frame::Inertial>&
        interface_unit_normal_one_form,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>&
        interface_unit_normal_vector,
    const tnsr::I<DataType, SpatialDim, Frame::Inertial>& shift,
    const DataType& used_for_size) {
  const auto outgoing_one_form =
      gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
          spacetime_normal_one_form, interface_unit_normal_one_form, shift, 1.);
  const auto incoming_one_form =
      gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
          spacetime_normal_one_form, interface_unit_normal_one_form, shift,
          -1.);
  const auto outgoing_vector =
      gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
          spacetime_normal_vector, interface_unit_normal_vector, 1.);
  const auto incoming_vector =
      gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
          spacetime_normal_vector, interface_unit_normal_vector, -1.);

  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(outgoing_one_form, outgoing_one_form,
                      inverse_spacetime_metric)),
      make_with_value<DataType>(used_for_size, 0.0), custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(incoming_one_form, incoming_one_form,
                      inverse_spacetime_metric)),
      make_with_value<DataType>(used_for_size, 0.0), custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(outgoing_vector, outgoing_vector, spacetime_metric)),
      make_with_value<DataType>(used_for_size, 0.0), custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(incoming_vector, incoming_vector, spacetime_metric)),
      make_with_value<DataType>(used_for_size, 0.0), custom_approx);

  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(outgoing_one_form, incoming_one_form,
                      inverse_spacetime_metric)),
      make_with_value<DataType>(used_for_size, -1.0), custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(outgoing_vector, incoming_vector, spacetime_metric)),
      make_with_value<DataType>(used_for_size, -1.0), custom_approx);

  const auto extended_interface_vector = extend_spatial_vector_to_spacetime(
      interface_unit_normal_vector, used_for_size);
  const auto extended_interface_one_form = extend_spatial_one_form_to_spacetime(
      interface_unit_normal_one_form, shift, used_for_size);

  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(extended_interface_vector, outgoing_one_form)),
      make_with_value<DataType>(used_for_size, 1.0 / std::numbers::sqrt2),
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(extended_interface_vector, incoming_one_form)),
      make_with_value<DataType>(used_for_size, -1.0 / std::numbers::sqrt2),
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(extended_interface_one_form, outgoing_vector)),
      make_with_value<DataType>(used_for_size, 1.0 / std::numbers::sqrt2),
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(extended_interface_one_form, incoming_vector)),
      make_with_value<DataType>(used_for_size, -1.0 / std::numbers::sqrt2),
      custom_approx);
}

template <size_t SpatialDim, typename DataType>
void test_null_normal_contractions_random_background(
    const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  const auto lapse =
      TestHelpers::gr::random_lapse(make_not_null(&generator), used_for_size);
  const auto shift = TestHelpers::gr::random_shift<SpatialDim>(
      make_not_null(&generator), used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<SpatialDim>(
          make_not_null(&generator), used_for_size);

  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<DataType, SpatialDim, Frame::Inertial>(
          lapse);
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);

  const auto interface_unit_normal_vector =
      random_unit_normal(make_not_null(&generator), spatial_metric);
  auto interface_unit_normal_one_form =
      make_with_value<tnsr::i<DataType, SpatialDim, Frame::Inertial>>(
          used_for_size, 0.0);
  raise_or_lower_index(make_not_null(&interface_unit_normal_one_form),
                       interface_unit_normal_vector, spatial_metric);

  check_null_normal_contractions(
      spacetime_metric, inverse_spacetime_metric, spacetime_normal_one_form,
      spacetime_normal_vector, interface_unit_normal_one_form,
      interface_unit_normal_vector, shift, used_for_size);
}

template <typename DataType>
void test_null_normal_contractions_kerr(const DataType& used_for_size) {
  constexpr size_t spatial_dim = 3;
  const gr::Solutions::KerrSchild kerr_solution{
      1.7, {{0.2, -0.4, 0.3}}, {{0.1, -0.2, 0.3}}};
  MAKE_GENERATOR(generator);
  const auto coords =
      make_random_vector_in_magnitude_range_flat<DataType, spatial_dim,
                                                 UpLo::Up, Frame::Inertial>(
          make_not_null(&generator), used_for_size, 4.0, 6.0);

  const auto vars = kerr_solution.variables(
      coords, 0.0,
      tmpl::list<
          gr::Tags::Lapse<DataType>,
          gr::Tags::Shift<DataType, spatial_dim, Frame::Inertial>,
          gr::Tags::SpatialMetric<DataType, spatial_dim, Frame::Inertial>,
          gr::Tags::InverseSpatialMetric<DataType, spatial_dim,
                                         Frame::Inertial>>{});
  const auto& lapse = get<gr::Tags::Lapse<DataType>>(vars);
  const auto& shift =
      get<gr::Tags::Shift<DataType, spatial_dim, Frame::Inertial>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, spatial_dim, Frame::Inertial>>(
          vars);
  const auto& inverse_spatial_metric = get<
      gr::Tags::InverseSpatialMetric<DataType, spatial_dim, Frame::Inertial>>(
      vars);
  const auto spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);

  const auto interface_unit_normal_vector =
      random_unit_normal(make_not_null(&generator), spatial_metric);
  auto interface_unit_normal_one_form =
      make_with_value<tnsr::i<DataType, spatial_dim, Frame::Inertial>>(
          used_for_size, 0.0);
  raise_or_lower_index(make_not_null(&interface_unit_normal_one_form),
                       interface_unit_normal_vector, spatial_metric);

  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<DataType, spatial_dim, Frame::Inertial>(
          lapse);
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);
  check_null_normal_contractions(
      spacetime_metric, inverse_spacetime_metric, spacetime_normal_one_form,
      spacetime_normal_vector, interface_unit_normal_one_form,
      interface_unit_normal_vector, shift, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.IntfcNullNormals",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_interface_null_normals, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_one_form_time_component_difference,
                                    ());
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(
      test_null_normal_contractions_random_background, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_null_normal_contractions_kerr, ());
}
