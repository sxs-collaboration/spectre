// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "tests/Unit/PointwiseFunctions/GeneralRelativity/GrTestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_1d_spatial_raise(const DataVector& used_for_size) {
  const size_t dim = 1;
  const tnsr::Ijj<double, dim> raised_tensor = raise_or_lower_first_index(
      make_deriv_spatial_metric<dim>(0.), make_inverse_spatial_metric<dim>(0.));

  CHECK(raised_tensor.get(0, 0, 0) == approx(3.));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_first_index(
          make_deriv_spatial_metric<dim>(used_for_size),
          make_inverse_spatial_metric<dim>(used_for_size)),
      raised_tensor);

  const tnsr::I<double, dim> shift = raise_or_lower_index(
      make_lower_shift<dim>(0.), make_inverse_spatial_metric<dim>(0.));
  CHECK(get<0>(shift) == approx(1.));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_index(make_lower_shift<dim>(used_for_size),
                           make_inverse_spatial_metric<dim>(used_for_size)),
      shift);
}

void test_1d_spatial_lower(const DataVector& used_for_size) {
  const size_t dim = 1;
  const tnsr::ijj<double, dim> lowered_tensor =
      raise_or_lower_first_index(make_spatial_christoffel_second_kind<dim>(0.),
                                 make_spatial_metric<dim>(0.));

  CHECK(lowered_tensor.get(0, 0, 0) == approx(4.));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_first_index(
          make_spatial_christoffel_second_kind<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size)),
      lowered_tensor);

  const tnsr::i<double, dim> lowered_shift =
      raise_or_lower_index(make_shift<dim>(0.), make_spatial_metric<dim>(0.));
  CHECK(get<0>(lowered_shift) == approx(1.));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_index(make_shift<dim>(used_for_size),
                           make_spatial_metric<dim>(used_for_size)),
      lowered_shift);
}

void test_2d_spatial_raise(const DataVector& used_for_size) {
  const size_t dim = 2;

  const tnsr::Ijj<double, dim> raised_tensor = raise_or_lower_first_index(
      make_deriv_spatial_metric<dim>(0.), make_inverse_spatial_metric<dim>(0.));

  CHECK(raised_tensor.get(0, 0, 0) == approx(11));
  CHECK(raised_tensor.get(0, 0, 1) == approx(20));
  CHECK(raised_tensor.get(0, 1, 1) == approx(38));
  CHECK(raised_tensor.get(1, 0, 0) == approx(22));
  CHECK(raised_tensor.get(1, 0, 1) == approx(40));
  CHECK(raised_tensor.get(1, 1, 1) == approx(76));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_first_index(
          make_deriv_spatial_metric<dim>(used_for_size),
          make_inverse_spatial_metric<dim>(used_for_size)),
      raised_tensor);

  const tnsr::I<double, dim> shift = raise_or_lower_index(
      make_lower_shift<dim>(0.), make_inverse_spatial_metric<dim>(0.));
  CHECK(get<0>(shift) == approx(5.));
  CHECK(get<1>(shift) == approx(10.));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_index(make_lower_shift<dim>(used_for_size),
                           make_inverse_spatial_metric<dim>(used_for_size)),
      shift);
}

void test_2d_spatial_lower(const DataVector& used_for_size) {
  const size_t dim = 2;

  const tnsr::ijj<double, dim> lowered_tensor =
      raise_or_lower_first_index(make_spatial_christoffel_second_kind<dim>(0.),
                                 make_spatial_metric<dim>(0.));

  CHECK(lowered_tensor.get(0, 0, 0) == approx(10));
  CHECK(lowered_tensor.get(0, 0, 1) == approx(22));
  CHECK(lowered_tensor.get(0, 1, 1) == approx(46));
  CHECK(lowered_tensor.get(1, 0, 0) == approx(20));
  CHECK(lowered_tensor.get(1, 0, 1) == approx(44));
  CHECK(lowered_tensor.get(1, 1, 1) == approx(92));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_first_index(
          make_spatial_christoffel_second_kind<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size)),
      lowered_tensor);

  const tnsr::i<double, dim> lowered_shift =
      raise_or_lower_index(make_shift<dim>(0.), make_spatial_metric<dim>(0.));
  CHECK(get<0>(lowered_shift) == approx(5.));
  CHECK(get<1>(lowered_shift) == approx(10.));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_index(make_shift<dim>(used_for_size),
                           make_spatial_metric<dim>(used_for_size)),
      lowered_shift);
}

void test_3d_spatial_raise(const DataVector& used_for_size) {
  const size_t dim = 3;
  const tnsr::Ijj<double, dim> raised_tensor = raise_or_lower_first_index(
      make_deriv_spatial_metric<dim>(0.), make_inverse_spatial_metric<dim>(0.));

  CHECK(raised_tensor.get(0, 0, 0) == approx(26));
  CHECK(raised_tensor.get(0, 0, 1) == approx(44));
  CHECK(raised_tensor.get(0, 0, 2) == approx(62));
  CHECK(raised_tensor.get(0, 1, 1) == approx(80));
  CHECK(raised_tensor.get(0, 1, 2) == approx(116));
  CHECK(raised_tensor.get(0, 2, 2) == approx(170));
  CHECK(raised_tensor.get(1, 0, 0) == approx(52));
  CHECK(raised_tensor.get(1, 0, 1) == approx(88));
  CHECK(raised_tensor.get(1, 0, 2) == approx(124));
  CHECK(raised_tensor.get(1, 1, 1) == approx(160));
  CHECK(raised_tensor.get(1, 1, 2) == approx(232));
  CHECK(raised_tensor.get(1, 2, 2) == approx(340));
  CHECK(raised_tensor.get(2, 0, 0) == approx(78));
  CHECK(raised_tensor.get(2, 0, 1) == approx(132));
  CHECK(raised_tensor.get(2, 0, 2) == approx(186));
  CHECK(raised_tensor.get(2, 1, 1) == approx(240));
  CHECK(raised_tensor.get(2, 1, 2) == approx(348));
  CHECK(raised_tensor.get(2, 2, 2) == approx(510));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_first_index(
          make_deriv_spatial_metric<dim>(used_for_size),
          make_inverse_spatial_metric<dim>(used_for_size)),
      raised_tensor);

  const tnsr::I<double, dim> shift = raise_or_lower_index(
      make_lower_shift<dim>(0.), make_inverse_spatial_metric<dim>(0.));
  CHECK(get<0>(shift) == approx(14.));
  CHECK(get<1>(shift) == approx(28.));
  CHECK(get<2>(shift) == approx(42.));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_index(make_lower_shift<dim>(used_for_size),
                           make_inverse_spatial_metric<dim>(used_for_size)),
      shift);
}

void test_3d_spatial_lower(const DataVector& used_for_size) {
  const size_t dim = 3;
  const tnsr::ijj<double, dim> lowered_tensor =
      raise_or_lower_first_index(make_spatial_christoffel_second_kind<dim>(0.),
                                 make_spatial_metric<dim>(0.));

  CHECK(lowered_tensor.get(0, 0, 0) == approx(16));
  CHECK(lowered_tensor.get(0, 0, 1) == approx(40));
  CHECK(lowered_tensor.get(0, 0, 2) == approx(64));
  CHECK(lowered_tensor.get(0, 1, 1) == approx(88));
  CHECK(lowered_tensor.get(0, 1, 2) == approx(136));
  CHECK(lowered_tensor.get(0, 2, 2) == approx(208));
  CHECK(lowered_tensor.get(1, 0, 0) == approx(32));
  CHECK(lowered_tensor.get(1, 0, 1) == approx(80));
  CHECK(lowered_tensor.get(1, 0, 2) == approx(128));
  CHECK(lowered_tensor.get(1, 1, 1) == approx(176));
  CHECK(lowered_tensor.get(1, 1, 2) == approx(272));
  CHECK(lowered_tensor.get(1, 2, 2) == approx(416));
  CHECK(lowered_tensor.get(2, 0, 0) == approx(48));
  CHECK(lowered_tensor.get(2, 0, 1) == approx(120));
  CHECK(lowered_tensor.get(2, 0, 2) == approx(192));
  CHECK(lowered_tensor.get(2, 1, 1) == approx(264));
  CHECK(lowered_tensor.get(2, 1, 2) == approx(408));
  CHECK(lowered_tensor.get(2, 2, 2) == approx(624));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_first_index(
          make_spatial_christoffel_second_kind<dim>(used_for_size),
          make_spatial_metric<dim>(used_for_size)),
      lowered_tensor);

  const tnsr::i<double, dim> lowered_shift =
      raise_or_lower_index(make_shift<dim>(0.), make_spatial_metric<dim>(0.));
  CHECK(get<0>(lowered_shift) == approx(14.));
  CHECK(get<1>(lowered_shift) == approx(28.));
  CHECK(get<2>(lowered_shift) == approx(42.));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_index(make_shift<dim>(used_for_size),
                           make_spatial_metric<dim>(used_for_size)),
      lowered_shift);
}

void test_3d_spacetime_raise(const DataVector& used_for_size) {
  const size_t dim = 3;
  const tnsr::Abb<double, dim> raised_tensor =
      raise_or_lower_first_index(make_spacetime_deriv_spacetime_metric<dim>(0.),
                                 make_inverse_spacetime_metric<dim>(0.));

  CHECK(raised_tensor.get(0, 0, 0) == approx(-272));
  CHECK(raised_tensor.get(0, 0, 1) == approx(-544));
  CHECK(raised_tensor.get(0, 0, 2) == approx(-816));
  CHECK(raised_tensor.get(0, 0, 3) == approx(-1088));
  CHECK(raised_tensor.get(0, 1, 1) == approx(-1088));
  CHECK(raised_tensor.get(0, 1, 2) == approx(-1632));
  CHECK(raised_tensor.get(0, 1, 3) == approx(-2176));
  CHECK(raised_tensor.get(0, 2, 2) == approx(-2448));
  CHECK(raised_tensor.get(0, 2, 3) == approx(-3264));
  CHECK(raised_tensor.get(0, 3, 3) == approx(-4352));
  CHECK(raised_tensor.get(1, 0, 0) == approx(-408));
  CHECK(raised_tensor.get(1, 0, 1) == approx(-816));
  CHECK(raised_tensor.get(1, 0, 2) == approx(-1224));
  CHECK(raised_tensor.get(1, 0, 3) == approx(-1632));
  CHECK(raised_tensor.get(1, 1, 1) == approx(-1632));
  CHECK(raised_tensor.get(1, 1, 2) == approx(-2448));
  CHECK(raised_tensor.get(1, 1, 3) == approx(-3264));
  CHECK(raised_tensor.get(1, 2, 2) == approx(-3672));
  CHECK(raised_tensor.get(1, 2, 3) == approx(-4896));
  CHECK(raised_tensor.get(1, 3, 3) == approx(-6528));
  CHECK(raised_tensor.get(2, 0, 0) == approx(-544));
  CHECK(raised_tensor.get(2, 0, 1) == approx(-1088));
  CHECK(raised_tensor.get(2, 0, 2) == approx(-1632));
  CHECK(raised_tensor.get(2, 0, 3) == approx(-2176));
  CHECK(raised_tensor.get(2, 1, 1) == approx(-2176));
  CHECK(raised_tensor.get(2, 1, 2) == approx(-3264));
  CHECK(raised_tensor.get(2, 1, 3) == approx(-4352));
  CHECK(raised_tensor.get(2, 2, 2) == approx(-4896));
  CHECK(raised_tensor.get(2, 2, 3) == approx(-6528));
  CHECK(raised_tensor.get(2, 3, 3) == approx(-8704));
  CHECK(raised_tensor.get(3, 0, 0) == approx(-680));
  CHECK(raised_tensor.get(3, 0, 1) == approx(-1360));
  CHECK(raised_tensor.get(3, 0, 2) == approx(-2040));
  CHECK(raised_tensor.get(3, 0, 3) == approx(-2720));
  CHECK(raised_tensor.get(3, 1, 1) == approx(-2720));
  CHECK(raised_tensor.get(3, 1, 2) == approx(-4080));
  CHECK(raised_tensor.get(3, 1, 3) == approx(-5440));
  CHECK(raised_tensor.get(3, 2, 2) == approx(-6120));
  CHECK(raised_tensor.get(3, 2, 3) == approx(-8160));
  CHECK(raised_tensor.get(3, 3, 3) == approx(-10880));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_first_index(
          make_spacetime_deriv_spacetime_metric<dim>(used_for_size),
          make_inverse_spacetime_metric<dim>(used_for_size)),
      raised_tensor);

  const tnsr::A<double, dim> raised_one_form = raise_or_lower_index(
      make_dummy_one_form<dim>(0.), make_inverse_spacetime_metric<dim>(0.));
  CHECK(get<0>(raised_one_form) == approx(-160.));
  CHECK(get<1>(raised_one_form) == approx(-240.));
  CHECK(get<2>(raised_one_form) == approx(-320.));
  CHECK(get<3>(raised_one_form) == approx(-400.));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_index(make_dummy_one_form<dim>(used_for_size),
                           make_inverse_spacetime_metric<dim>(used_for_size)),
      raised_one_form);
}

void test_3d_spacetime_lower(const DataVector& used_for_size) {
  const size_t dim = 3;
  const tnsr::abb<double, dim> lowered_tensor = raise_or_lower_first_index(
      make_spacetime_christoffel_second_kind<dim>(0.),
      make_spacetime_metric<dim>(0.));

  CHECK(lowered_tensor.get(0, 0, 0) == approx(-432));
  CHECK(lowered_tensor.get(0, 0, 1) == approx(-864));
  CHECK(lowered_tensor.get(0, 0, 2) == approx(-1296));
  CHECK(lowered_tensor.get(0, 0, 3) == approx(-1728));
  CHECK(lowered_tensor.get(0, 1, 1) == approx(-1296));
  CHECK(lowered_tensor.get(0, 1, 2) == approx(-1944));
  CHECK(lowered_tensor.get(0, 1, 3) == approx(-2592));
  CHECK(lowered_tensor.get(0, 2, 2) == approx(-2592));
  CHECK(lowered_tensor.get(0, 2, 3) == approx(-3456));
  CHECK(lowered_tensor.get(0, 3, 3) == approx(-4320));
  CHECK(lowered_tensor.get(1, 0, 0) == approx(-648));
  CHECK(lowered_tensor.get(1, 0, 1) == approx(-1296));
  CHECK(lowered_tensor.get(1, 0, 2) == approx(-1944));
  CHECK(lowered_tensor.get(1, 0, 3) == approx(-2592));
  CHECK(lowered_tensor.get(1, 1, 1) == approx(-1944));
  CHECK(lowered_tensor.get(1, 1, 2) == approx(-2916));
  CHECK(lowered_tensor.get(1, 1, 3) == approx(-3888));
  CHECK(lowered_tensor.get(1, 2, 2) == approx(-3888));
  CHECK(lowered_tensor.get(1, 2, 3) == approx(-5184));
  CHECK(lowered_tensor.get(1, 3, 3) == approx(-6480));
  CHECK(lowered_tensor.get(2, 0, 0) == approx(-864));
  CHECK(lowered_tensor.get(2, 0, 1) == approx(-1728));
  CHECK(lowered_tensor.get(2, 0, 2) == approx(-2592));
  CHECK(lowered_tensor.get(2, 0, 3) == approx(-3456));
  CHECK(lowered_tensor.get(2, 1, 1) == approx(-2592));
  CHECK(lowered_tensor.get(2, 1, 2) == approx(-3888));
  CHECK(lowered_tensor.get(2, 1, 3) == approx(-5184));
  CHECK(lowered_tensor.get(2, 2, 2) == approx(-5184));
  CHECK(lowered_tensor.get(2, 2, 3) == approx(-6912));
  CHECK(lowered_tensor.get(2, 3, 3) == approx(-8640));
  CHECK(lowered_tensor.get(3, 0, 0) == approx(-1080));
  CHECK(lowered_tensor.get(3, 0, 1) == approx(-2160));
  CHECK(lowered_tensor.get(3, 0, 2) == approx(-3240));
  CHECK(lowered_tensor.get(3, 0, 3) == approx(-4320));
  CHECK(lowered_tensor.get(3, 1, 1) == approx(-3240));
  CHECK(lowered_tensor.get(3, 1, 2) == approx(-4860));
  CHECK(lowered_tensor.get(3, 1, 3) == approx(-6480));
  CHECK(lowered_tensor.get(3, 2, 2) == approx(-6480));
  CHECK(lowered_tensor.get(3, 2, 3) == approx(-8640));
  CHECK(lowered_tensor.get(3, 3, 3) == approx(-10800));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_first_index(
          make_spacetime_christoffel_second_kind<dim>(used_for_size),
          make_spacetime_metric<dim>(used_for_size)),
      lowered_tensor);

  const tnsr::a<double, dim> lowered_vector = raise_or_lower_index(
      make_dummy_vector<dim>(0.), make_spacetime_metric<dim>(0.));
  CHECK(get<0>(lowered_vector) == approx(-160.));
  CHECK(get<1>(lowered_vector) == approx(-240.));
  CHECK(get<2>(lowered_vector) == approx(-320.));
  CHECK(get<3>(lowered_vector) == approx(-400.));

  check_tensor_doubles_equals_tensor_datavectors(
      raise_or_lower_index(make_dummy_vector<dim>(used_for_size),
                           make_spacetime_metric<dim>(used_for_size)),
      lowered_vector);
}
void test_1d_spatial_trace(const DataVector& used_for_size) {
  const size_t dim = 1;
  const tnsr::i<double, dim> covector = trace_last_indices(
      make_deriv_spatial_metric<dim>(0.), make_inverse_spatial_metric<dim>(0.));

  CHECK(covector.get(0) == approx(3.));

  check_tensor_doubles_equals_tensor_datavectors(
      trace_last_indices(make_deriv_spatial_metric<dim>(used_for_size),
                         make_inverse_spatial_metric<dim>(used_for_size)),
      covector);
}

void test_2d_spatial_trace(const DataVector& used_for_size) {
  const size_t dim = 2;
  const tnsr::i<double, dim> covector = trace_last_indices(
      make_deriv_spatial_metric<dim>(0.), make_inverse_spatial_metric<dim>(0.));

  CHECK(covector.get(0) == approx(75));
  CHECK(covector.get(1) == approx(84));

  check_tensor_doubles_equals_tensor_datavectors(
      trace_last_indices(make_deriv_spatial_metric<dim>(used_for_size),
                         make_inverse_spatial_metric<dim>(used_for_size)),
      covector);
}

void test_3d_spatial_trace(const DataVector& used_for_size) {
  const size_t dim = 3;
  const tnsr::i<double, dim> covector = trace_last_indices(
      make_deriv_spatial_metric<dim>(0.), make_inverse_spatial_metric<dim>(0.));

  CHECK(covector.get(0) == approx(588));
  CHECK(covector.get(1) == approx(624));
  CHECK(covector.get(2) == approx(660));

  check_tensor_doubles_equals_tensor_datavectors(
      trace_last_indices(make_deriv_spatial_metric<dim>(used_for_size),
                         make_inverse_spatial_metric<dim>(used_for_size)),
      covector);
}

void test_3d_spacetime_trace(const DataVector& used_for_size) {
  const size_t dim = 3;
  const tnsr::a<double, dim> vector =
      trace_last_indices(make_spacetime_deriv_spacetime_metric<dim>(0.),
                         make_inverse_spacetime_metric<dim>(0.));

  CHECK(vector.get(0) == approx(-9600));
  CHECK(vector.get(1) == approx(-12800));
  CHECK(vector.get(2) == approx(-16000));
  CHECK(vector.get(3) == approx(-19200));

  check_tensor_doubles_equals_tensor_datavectors(
      trace_last_indices(
          make_spacetime_deriv_spacetime_metric<dim>(used_for_size),
          make_inverse_spacetime_metric<dim>(used_for_size)),
      vector);
}
void test_1d_spatial_trace_tensor_type_aa(const DataVector& used_for_size) {
  const size_t dim = 1;
  const Scalar<double> scalar = trace(make_dt_spatial_metric<dim>(0.),
                                      make_inverse_spatial_metric<dim>(0.));

  CHECK(scalar.get() == approx(0.));

  check_tensor_doubles_equals_tensor_datavectors(
      trace(make_dt_spatial_metric<dim>(used_for_size),
            make_inverse_spatial_metric<dim>(used_for_size)),
      scalar);
}

void test_2d_spatial_trace_tensor_type_aa(const DataVector& used_for_size) {
  const size_t dim = 2;
  const Scalar<double> scalar = trace(make_dt_spatial_metric<dim>(0.),
                                      make_inverse_spatial_metric<dim>(0.));

  CHECK(scalar.get() == approx(12.));

  check_tensor_doubles_equals_tensor_datavectors(
      trace(make_dt_spatial_metric<dim>(used_for_size),
            make_inverse_spatial_metric<dim>(used_for_size)),
      scalar);
}

void test_3d_spatial_trace_tensor_type_aa(const DataVector& used_for_size) {
  const size_t dim = 3;
  const Scalar<double> scalar = trace(make_dt_spatial_metric<dim>(0.),
                                      make_inverse_spatial_metric<dim>(0.));

  CHECK(scalar.get() == approx(96.));

  check_tensor_doubles_equals_tensor_datavectors(
      trace(make_dt_spatial_metric<dim>(used_for_size),
            make_inverse_spatial_metric<dim>(used_for_size)),
      scalar);
}
void test_3d_spacetime_trace_tensor_type_aa(const DataVector& used_for_size) {
  const size_t dim = 3;
  const Scalar<double> scalar = trace(make_dt_spacetime_metric<dim>(0.),
                                      make_inverse_spacetime_metric<dim>(0.));

  CHECK(scalar.get() == approx(-1548.));

  check_tensor_doubles_equals_tensor_datavectors(
      trace(make_dt_spacetime_metric<dim>(used_for_size),
            make_inverse_spacetime_metric<dim>(used_for_size)),
      scalar);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.IndexManipulation",
                  "[PointwiseFunctions][Unit]") {
  const size_t dim = 3;
  const DataVector dv(2);

  test_1d_spatial_raise(dv);
  test_2d_spatial_raise(dv);
  test_3d_spatial_raise(dv);
  test_1d_spatial_lower(dv);
  test_2d_spatial_lower(dv);
  test_3d_spatial_lower(dv);
  test_3d_spacetime_raise(dv);
  test_3d_spacetime_lower(dv);
  test_1d_spatial_trace(dv);
  test_2d_spatial_trace(dv);
  test_3d_spatial_trace(dv);
  test_3d_spacetime_trace(dv);
  test_1d_spatial_trace_tensor_type_aa(dv);
  test_2d_spatial_trace_tensor_type_aa(dv);
  test_3d_spatial_trace_tensor_type_aa(dv);
  test_3d_spacetime_trace_tensor_type_aa(dv);
}
