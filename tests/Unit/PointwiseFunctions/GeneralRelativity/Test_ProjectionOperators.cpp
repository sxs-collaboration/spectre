// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/RandomUnitNormal.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/InterfaceNullNormal.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
const Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);

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
void check_aa_orthogonality(
    const tnsr::aa<DataType, SpatialDim, Frame::Inertial>& projection_aa,
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_vector) {
  const auto right_contraction = tenex::evaluate<ti::a>(
      projection_aa(ti::a, ti::b) * spacetime_normal_vector(ti::B));
  const auto left_contraction = tenex::evaluate<ti::b>(
      spacetime_normal_vector(ti::A) * projection_aa(ti::a, ti::b));
  const auto zero =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame::Inertial>>(
          get_size(get<0, 0>(projection_aa)), 0.0);
  CHECK_ITERABLE_CUSTOM_APPROX(right_contraction, zero, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(left_contraction, zero, custom_approx);
}

template <size_t SpatialDim, typename DataType>
void check_ab_orthogonality(
    const tnsr::Ab<DataType, SpatialDim, Frame::Inertial>& projection_ab,
    const tnsr::a<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame::Inertial>&
        spacetime_normal_vector) {
  const auto lower_contraction = tenex::evaluate<ti::b>(
      spacetime_normal_one_form(ti::a) * projection_ab(ti::A, ti::b));
  const auto upper_contraction = tenex::evaluate<ti::A>(
      projection_ab(ti::A, ti::b) * spacetime_normal_vector(ti::B));
  const auto zero_one_form =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame::Inertial>>(
          get_size(get<0, 0>(projection_ab)), 0.0);
  const auto zero_vector =
      make_with_value<tnsr::A<DataType, SpatialDim, Frame::Inertial>>(
          get_size(get<0, 0>(projection_ab)), 0.0);
  CHECK_ITERABLE_CUSTOM_APPROX(lower_contraction, zero_one_form, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(upper_contraction, zero_vector, custom_approx);
}

template <size_t SpatialDim, typename DataType>
void test_projection_operator(const DataType& used_for_size) {
  {
    tnsr::II<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::II<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(f, "ProjectionOperators",
                                      "transverse_projection_operator",
                                      {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::ii<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(f, "ProjectionOperators",
                                      "transverse_projection_operator",
                                      {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::Ij<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators",
        "transverse_projection_operator_mixed_from_spatial_input",
        {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::AA<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::AA<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::A<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators", "projection_operator_transverse_to_interface",
        {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::aa<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::aa<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::a<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators", "projection_operator_transverse_to_interface",
        {{{-1., 1.}}}, used_for_size);
  }

  {
    tnsr::Ab<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::A<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::a<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::i<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&) =
        &gr::transverse_projection_operator<DataType, SpatialDim,
                                            Frame::Inertial>;
    pypp::check_with_random_values<1>(
        f, "ProjectionOperators",
        "projection_operator_transverse_to_interface_mixed", {{{-1., 1.}}},
        used_for_size);
  }

  const auto data_size = get_size(used_for_size);
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

  const auto projection_aa = gr::transverse_projection_operator(
      spacetime_metric, spacetime_normal_one_form,
      interface_unit_normal_one_form, shift);
  check_aa_orthogonality(projection_aa, spacetime_normal_vector);

  tnsr::aa<DataType, SpatialDim, Frame::Inertial> projection_aa_not_null(
      data_size);
  gr::transverse_projection_operator(
      make_not_null(&projection_aa_not_null), spacetime_metric,
      spacetime_normal_one_form, interface_unit_normal_one_form, shift);
  check_aa_orthogonality(projection_aa_not_null, spacetime_normal_vector);

  const auto projection_ab = gr::transverse_projection_operator(
      spacetime_normal_vector, spacetime_normal_one_form,
      interface_unit_normal_vector, interface_unit_normal_one_form, shift);
  check_ab_orthogonality(projection_ab, spacetime_normal_one_form,
                         spacetime_normal_vector);

  tnsr::Ab<DataType, SpatialDim, Frame::Inertial> projection_ab_not_null(
      data_size);
  gr::transverse_projection_operator(
      make_not_null(&projection_ab_not_null), spacetime_normal_vector,
      spacetime_normal_one_form, interface_unit_normal_vector,
      interface_unit_normal_one_form, shift);
  check_ab_orthogonality(projection_ab_not_null, spacetime_normal_one_form,
                         spacetime_normal_vector);

  const auto outgoing_null_one_form =
      gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
          spacetime_normal_one_form, interface_unit_normal_one_form, shift,
          1.0);
  const auto incoming_null_one_form =
      gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
          spacetime_normal_one_form, interface_unit_normal_one_form, shift,
          -1.0);
  const auto outgoing_null_vector =
      gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
          spacetime_normal_vector, interface_unit_normal_vector, 1.0);
  const auto incoming_null_vector =
      gr::interface_null_normal<DataType, SpatialDim, Frame::Inertial>(
          spacetime_normal_vector, interface_unit_normal_vector, -1.0);

  const auto contract_aa_right = [&](const auto& vector) {
    return tenex::evaluate<ti::a>(projection_aa(ti::a, ti::b) * vector(ti::B));
  };
  const auto contract_aa_left = [&](const auto& vector) {
    return tenex::evaluate<ti::b>(vector(ti::A) * projection_aa(ti::a, ti::b));
  };
  const auto contract_ab_left = [&](const auto& one_form) {
    return tenex::evaluate<ti::b>(one_form(ti::a) *
                                  projection_ab(ti::A, ti::b));
  };
  const auto zero_spacetime_one_form =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame::Inertial>>(data_size,
                                                                      0.0);

  CHECK_ITERABLE_CUSTOM_APPROX(contract_aa_right(outgoing_null_vector),
                               zero_spacetime_one_form, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(contract_aa_right(incoming_null_vector),
                               zero_spacetime_one_form, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(contract_aa_left(outgoing_null_vector),
                               zero_spacetime_one_form, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(contract_aa_left(incoming_null_vector),
                               zero_spacetime_one_form, custom_approx);

  CHECK_ITERABLE_CUSTOM_APPROX(contract_ab_left(outgoing_null_one_form),
                               zero_spacetime_one_form, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(contract_ab_left(incoming_null_one_form),
                               zero_spacetime_one_form, custom_approx);

  const auto interface_normal_one_form = extend_spatial_one_form_to_spacetime(
      interface_unit_normal_one_form, shift, used_for_size);
  const auto interface_normal_vector = extend_spatial_vector_to_spacetime(
      interface_unit_normal_vector, used_for_size);

  CHECK_ITERABLE_CUSTOM_APPROX(contract_aa_right(interface_normal_vector),
                               zero_spacetime_one_form, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(contract_aa_left(interface_normal_vector),
                               zero_spacetime_one_form, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(contract_ab_left(interface_normal_one_form),
                               zero_spacetime_one_form, custom_approx);

  const auto trace_aa = tenex::evaluate(inverse_spacetime_metric(ti::A, ti::B) *
                                        projection_aa(ti::a, ti::b));
  const auto trace_ab = tenex::evaluate(projection_ab(ti::A, ti::a));
  const auto expected_trace =
      make_with_value<Scalar<DataType>>(used_for_size, SpatialDim - 1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(trace_aa, expected_trace, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(trace_ab, expected_trace, custom_approx);

  const auto raised_projection_aa = tenex::evaluate<ti::A, ti::b>(
      inverse_spacetime_metric(ti::A, ti::C) * projection_aa(ti::c, ti::b));
  CHECK_ITERABLE_CUSTOM_APPROX(raised_projection_aa, projection_ab,
                               custom_approx);

  const auto idempotent_projection_ab = tenex::evaluate<ti::A, ti::b>(
      projection_ab(ti::A, ti::c) * projection_ab(ti::C, ti::b));
  CHECK_ITERABLE_CUSTOM_APPROX(idempotent_projection_ab, projection_ab,
                               custom_approx);

  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(outgoing_null_one_form, outgoing_null_one_form,
                      inverse_spacetime_metric)),
      make_with_value<DataType>(used_for_size, 0.0), custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(incoming_null_one_form, incoming_null_one_form,
                      inverse_spacetime_metric)),
      make_with_value<DataType>(used_for_size, 0.0), custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(outgoing_null_vector, outgoing_null_vector,
                      spacetime_metric)),
      make_with_value<DataType>(used_for_size, 0.0), custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(dot_product(incoming_null_vector, incoming_null_vector,
                      spacetime_metric)),
      make_with_value<DataType>(used_for_size, 0.0), custom_approx);
}

template <size_t SpatialDim, typename DataType>
void test_projection_operator_spatial(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<SpatialDim>(
          make_not_null(&generator), used_for_size);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const auto interface_unit_normal_vector =
      random_unit_normal(make_not_null(&generator), spatial_metric);
  auto interface_unit_normal_one_form =
      make_with_value<tnsr::i<DataType, SpatialDim, Frame::Inertial>>(
          used_for_size, 0.0);
  raise_or_lower_index(make_not_null(&interface_unit_normal_one_form),
                       interface_unit_normal_vector, spatial_metric);

  const auto projection_II = gr::transverse_projection_operator(
      inverse_spatial_metric, interface_unit_normal_vector);
  const auto projection_ii = gr::transverse_projection_operator(
      spatial_metric, interface_unit_normal_one_form);
  const auto projection_Ij = gr::transverse_projection_operator(
      interface_unit_normal_vector, interface_unit_normal_one_form);

  auto projection_II_not_null =
      make_with_value<tnsr::II<DataType, SpatialDim, Frame::Inertial>>(
          used_for_size, 0.0);
  gr::transverse_projection_operator(make_not_null(&projection_II_not_null),
                                     inverse_spatial_metric,
                                     interface_unit_normal_vector);
  CHECK_ITERABLE_CUSTOM_APPROX(projection_II_not_null, projection_II,
                               custom_approx);

  auto projection_ii_not_null =
      make_with_value<tnsr::ii<DataType, SpatialDim, Frame::Inertial>>(
          used_for_size, 0.0);
  gr::transverse_projection_operator(make_not_null(&projection_ii_not_null),
                                     spatial_metric,
                                     interface_unit_normal_one_form);
  CHECK_ITERABLE_CUSTOM_APPROX(projection_ii_not_null, projection_ii,
                               custom_approx);

  auto projection_Ij_not_null =
      make_with_value<tnsr::Ij<DataType, SpatialDim, Frame::Inertial>>(
          used_for_size, 0.0);
  gr::transverse_projection_operator(make_not_null(&projection_Ij_not_null),
                                     interface_unit_normal_vector,
                                     interface_unit_normal_one_form);
  CHECK_ITERABLE_CUSTOM_APPROX(projection_Ij_not_null, projection_Ij,
                               custom_approx);

  const auto zero_vector =
      make_with_value<tnsr::I<DataType, SpatialDim, Frame::Inertial>>(
          used_for_size, 0.0);
  const auto zero_one_form =
      make_with_value<tnsr::i<DataType, SpatialDim, Frame::Inertial>>(
          used_for_size, 0.0);

  CHECK_ITERABLE_CUSTOM_APPROX(
      tenex::evaluate<ti::I>(projection_II(ti::I, ti::J) *
                             interface_unit_normal_one_form(ti::j)),
      zero_vector, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      tenex::evaluate<ti::J>(interface_unit_normal_one_form(ti::i) *
                             projection_II(ti::I, ti::J)),
      zero_vector, custom_approx);

  CHECK_ITERABLE_CUSTOM_APPROX(
      tenex::evaluate<ti::i>(projection_ii(ti::i, ti::j) *
                             interface_unit_normal_vector(ti::J)),
      zero_one_form, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      tenex::evaluate<ti::j>(interface_unit_normal_vector(ti::I) *
                             projection_ii(ti::i, ti::j)),
      zero_one_form, custom_approx);

  CHECK_ITERABLE_CUSTOM_APPROX(
      tenex::evaluate<ti::I>(projection_Ij(ti::I, ti::j) *
                             interface_unit_normal_vector(ti::J)),
      zero_vector, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      tenex::evaluate<ti::j>(interface_unit_normal_one_form(ti::i) *
                             projection_Ij(ti::I, ti::j)),
      zero_one_form, custom_approx);

  const auto trace_II = tenex::evaluate(spatial_metric(ti::i, ti::j) *
                                        projection_II(ti::I, ti::J));
  const auto trace_ii = tenex::evaluate(inverse_spatial_metric(ti::I, ti::J) *
                                        projection_ii(ti::i, ti::j));
  const auto trace_Ij = tenex::evaluate(projection_Ij(ti::I, ti::i));
  const auto expected_trace =
      make_with_value<Scalar<DataType>>(used_for_size, SpatialDim - 1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(trace_II, expected_trace, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(trace_ii, expected_trace, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(trace_Ij, expected_trace, custom_approx);

  const auto idempotent_projection_Ij = tenex::evaluate<ti::I, ti::j>(
      projection_Ij(ti::I, ti::k) * projection_Ij(ti::K, ti::j));
  CHECK_ITERABLE_CUSTOM_APPROX(idempotent_projection_Ij, projection_Ij,
                               custom_approx);

  const auto raised_projection_ii = tenex::evaluate<ti::I, ti::j>(
      inverse_spatial_metric(ti::I, ti::K) * projection_ii(ti::k, ti::j));
  CHECK_ITERABLE_CUSTOM_APPROX(raised_projection_ii, projection_Ij,
                               custom_approx);
}
}  // namespace

namespace {
using frame = Frame::Inertial;
constexpr size_t SpatialDim = 3;

// Compare with fixed reference values from SpEC on a
// 3D mesh with 3x3x3 grid points.
void compare_spatial_projection_tensors_with_spec() {
  constexpr size_t grid_size_each_dimension = 3;
  const std::array<double, 3> lower_bound{{299., -0.5, -0.5}};
  // Setup grid
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  // Setup coordinates
  const Direction<SpatialDim> direction(1, Side::Upper);  // +y direction
  const size_t slice_grid_points =
      mesh.extents().slice_away(direction.dimension()).product();
  const auto inertial_coords = [&slice_grid_points, &lower_bound]() {
    tnsr::I<DataVector, SpatialDim, frame> tmp(slice_grid_points, 0.);
    // +y direction
    get<1>(tmp) = 0.5;
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        get<0>(tmp)[i * SpatialDim + j] =
            lower_bound[0] + 0.5 * static_cast<double>(i);
        get<2>(tmp)[i * SpatialDim + j] =
            lower_bound[2] + 0.5 * static_cast<double>(j);
      }
    }
    return tmp;
  }();

  // 1. Projection IJ
  auto local_inverse_spatial_metric =
      make_with_value<tnsr::II<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_unit_interface_normal_vector =
      make_with_value<tnsr::I<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_spatial_projection_IJ =
      make_with_value<tnsr::II<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);

  // Setting inverse_spatial_metric to compare with values from SpEC
  for (size_t i = 0; i < get<0>(inertial_coords).size(); ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      local_inverse_spatial_metric.get(0, j)[i] = 41.;
      local_inverse_spatial_metric.get(1, j)[i] = 43.;
      local_inverse_spatial_metric.get(2, j)[i] = 47.;
    }
  }
  // Setting unit_interface_normal_vector to compare with values from SpEC
  get<0>(local_unit_interface_normal_vector) = -1.;
  get<1>(local_unit_interface_normal_vector) = 1.;
  get<2>(local_unit_interface_normal_vector) = 1.;

  // Call tested function
  gr::transverse_projection_operator(
      make_not_null(&local_spatial_projection_IJ), local_inverse_spatial_metric,
      local_unit_interface_normal_vector);

  // Initialize with values from SpEC
  auto spec_spatial_projection_IJ =
      make_with_value<tnsr::II<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  {
    const std::array<double, 9> spec_vals = {
        {40., 42., 42., 42., 42., 42., 42., 42., 46.}};
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = j; k < SpatialDim; ++k) {
        spec_spatial_projection_IJ.get(j, k) =
            gsl::at(spec_vals, j * SpatialDim + k);
      }
    }
  }

  // Compare values returned to those from SpEC
  CHECK_ITERABLE_CUSTOM_APPROX(local_spatial_projection_IJ,
                               spec_spatial_projection_IJ, custom_approx);

  // 2. Projection ij
  auto local_spatial_metric =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_unit_interface_normal_one_form =
      make_with_value<tnsr::i<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  auto local_spatial_projection_ij =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);

  // Setting inverse_spatial_metric to compare with values from SpEC
  for (size_t i = 0; i < SpatialDim; ++i) {
    local_spatial_metric.get(0, i) = 263.;
    local_spatial_metric.get(1, i) = 269.;
    local_spatial_metric.get(2, i) = 271.;
  }
  // Setting unit_interface_normal_vector to compare with values from SpEC
  get<0>(local_unit_interface_normal_one_form) = -1.;
  get<1>(local_unit_interface_normal_one_form) = 1.;
  get<2>(local_unit_interface_normal_one_form) = 1.;

  // Call tested function
  gr::transverse_projection_operator(
      make_not_null(&local_spatial_projection_ij), local_spatial_metric,
      local_unit_interface_normal_one_form);

  // Initialize with values from SpEC
  auto spec_spatial_projection_ij =
      make_with_value<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  {
    const std::array<double, 9> spec_vals = {
        {262., 264., 264., 264., 268., 268., 264., 268., 270.}};
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = j; k < SpatialDim; ++k) {
        spec_spatial_projection_ij.get(j, k) =
            gsl::at(spec_vals, j * SpatialDim + k);
      }
    }
  }

  // Compare values returned to those from SpEC
  CHECK_ITERABLE_CUSTOM_APPROX(local_spatial_projection_ij,
                               spec_spatial_projection_ij, custom_approx);

  // 3. Projection Ij
  auto local_spatial_projection_Ij =
      make_with_value<tnsr::Ij<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);

  // Call tested function
  gr::transverse_projection_operator(
      make_not_null(&local_spatial_projection_Ij),
      local_unit_interface_normal_vector, local_unit_interface_normal_one_form);

  // Initialize with values from SpEC
  auto spec_spatial_projection_Ij =
      make_with_value<tnsr::Ij<DataVector, SpatialDim, Frame::Inertial>>(
          inertial_coords, 0.);
  {
    const std::array<double, 9> spec_vals = {
        {0., 1., 1., 1., 0., -1., 1., -1., 0.}};
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        spec_spatial_projection_Ij.get(j, k) =
            gsl::at(spec_vals, j * SpatialDim + k);
      }
    }
  }

  // Compare values returned to those from SpEC
  CHECK_ITERABLE_CUSTOM_APPROX(local_spatial_projection_Ij,
                               spec_spatial_projection_Ij, custom_approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.ProjectionOps",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_projection_operator, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_projection_operator_spatial,
                                    (1, 2, 3));

  compare_spatial_projection_tensors_with_spec();
}
