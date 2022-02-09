// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/DetAndInverseSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpatialMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace Tags {
template <typename Tag, typename Dim, typename Frame, typename>
struct deriv;
}  // namespace Tags

namespace {
template <size_t Dim, typename DataType>
void test_compute_spacetime_metric(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::aa<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&)>(
          &gr::spacetime_metric<Dim, Frame::Inertial, DataType>),
      "ComputeSpacetimeQuantities", "spacetime_metric", {{{-10., 10.}}},
      used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_inverse_spacetime_metric(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::AA<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::II<DataType, Dim, Frame::Inertial>&)>(
          &gr::inverse_spacetime_metric<Dim, Frame::Inertial, DataType>),
      "ComputeSpacetimeQuantities", "inverse_spacetime_metric", {{{-10., 10.}}},
      used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_time_derivative_of_spacetime_metric(
    const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::aa<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&, const Scalar<DataType>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&)>(
          &gr::time_derivative_of_spacetime_metric<Dim, Frame::Inertial,
                                                   DataType>),
      "ComputeSpacetimeQuantities", "dt_spacetime_metric", {{{-10., 10.}}},
      used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_time_derivative_of_spatial_metric(
    const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ii<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::iJ<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::ijj<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&)>(
          &gr::time_derivative_of_spatial_metric<Dim, Frame::Inertial,
                                                 DataType>),
      "ComputeSpacetimeQuantities", "dt_spatial_metric", {{{-10., 10.}}},
      used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_derivatives_of_spacetime_metric(
    const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::abb<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&, const Scalar<DataType>&,
          const tnsr::i<DataType, Dim, Frame::Inertial>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::iJ<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::ijj<DataType, Dim, Frame::Inertial>&)>(
          &gr::derivatives_of_spacetime_metric<Dim, Frame::Inertial, DataType>),
      "ComputeSpacetimeQuantities", "derivatives_of_spacetime_metric",
      {{{-10., 10.}}}, used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_spacetime_normal_vector(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::A<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&)>(
          &gr::spacetime_normal_vector<Dim, Frame::Inertial, DataType>),
      "ComputeSpacetimeQuantities", "spacetime_normal_vector", {{{-10., 10.}}},
      used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_spacetime_normal_one_form(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1., 1.);
  const auto lapse = make_with_random_values<Scalar<DataType>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<Dim, Frame::Inertial>(lapse);
  CHECK_ITERABLE_APPROX(spacetime_normal_one_form.get(0), -lapse.get());
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(spacetime_normal_one_form.get(i + 1),
                          make_with_value<DataType>(used_for_size, 0.));
  }
}
template <size_t Dim, typename DataType>
void test_compute_extrinsic_curvature(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::ii<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::iJ<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::ijj<DataType, Dim, Frame::Inertial>&)>(
          &gr::extrinsic_curvature<Dim, Frame::Inertial, DataType>),
      "ComputeSpacetimeQuantities", "extrinsic_curvature", {{{-10., 10.}}},
      used_for_size);
}

template <size_t Dim, typename T>
void test_compute_spatial_metric_lapse_shift(const T& used_for_size) {
  // Set up random values for lapse, shift, and spatial_metric.
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  std::uniform_real_distribution<> dist_positive(1., 2.);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);
  const auto nn_dist_positive = make_not_null(&dist_positive);

  const auto lapse = make_with_random_values<Scalar<T>>(
      nn_generator, nn_dist_positive, used_for_size);
  const auto shift = make_with_random_values<tnsr::I<T, Dim>>(
      nn_generator, nn_dist, used_for_size);
  const auto spatial_metric = [&]() {
    auto spatial_metric_l = make_with_random_values<tnsr::ii<T, Dim>>(
        nn_generator, nn_dist, used_for_size);
    // Make sure spatial_metric isn't singular by adding
    // large enough positive diagonal values.
    for (size_t i = 0; i < Dim; ++i) {
      spatial_metric_l.get(i, i) += 4.;
    }
    return spatial_metric_l;
  }();

  // Make spacetime metric from spatial metric, lapse, and shift.
  // Then go backwards and compute the spatial metric, lapse, and shift
  // and make sure we get back the original values.
  const auto psi = gr::spacetime_metric(lapse, shift, spatial_metric);

  // Here are the functions we are testing.
  const auto spatial_metric_test = gr::spatial_metric(psi);
  const auto shift_test =
      gr::shift(psi, determinant_and_inverse(spatial_metric).second);
  const auto lapse_test = gr::lapse(shift, psi);

  CHECK_ITERABLE_APPROX(spatial_metric, spatial_metric_test);
  CHECK_ITERABLE_APPROX(shift, shift_test);
  CHECK_ITERABLE_APPROX(lapse, lapse_test);
}

template <size_t Dim, typename DataType>
void test_compute_deriv_inverse_spatial_metric(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iJJ<DataType, Dim, Frame::Inertial> (*)(
          const tnsr::II<DataType, Dim, Frame::Inertial>&,
          const tnsr::ijj<DataType, Dim, Frame::Inertial>&)>(
          &gr::deriv_inverse_spatial_metric<Dim, Frame::Inertial, DataType>),
      "ComputeSpacetimeQuantities", "deriv_inverse_spatial_metric",
      {{{-10., 10.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.SpacetimeDecomp",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_spacetime_metric, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_inverse_spacetime_metric,
                                    (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(
      test_compute_derivatives_of_spacetime_metric, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(
      test_compute_time_derivative_of_spacetime_metric, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(
      test_compute_time_derivative_of_spatial_metric, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_spacetime_normal_vector,
                                    (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_spacetime_normal_one_form,
                                    (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_spatial_metric_lapse_shift,
                                    (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_extrinsic_curvature,
                                    (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_deriv_inverse_spatial_metric,
                                    (1, 2, 3));

  // Check that compute items work correctly in the DataBox
  // First, check that the names are correct
  TestHelpers::db::test_compute_tag<
      gr::Tags::SpacetimeNormalOneFormCompute<3, Frame::Inertial, DataVector>>(
      "SpacetimeNormalOneForm");
  TestHelpers::db::test_compute_tag<
      gr::Tags::SpacetimeNormalVectorCompute<3, Frame::Inertial, DataVector>>(
      "SpacetimeNormalVector");
  TestHelpers::db::test_compute_tag<
      gr::Tags::SpacetimeMetricCompute<3, Frame::Inertial, DataVector>>(
      "SpacetimeMetric");
  TestHelpers::db::test_compute_tag<
      gr::Tags::InverseSpacetimeMetricCompute<3, Frame::Inertial, DataVector>>(
      "InverseSpacetimeMetric");
  TestHelpers::db::test_compute_tag<
      gr::Tags::SpatialMetricCompute<3, Frame::Inertial, DataVector>>(
      "SpatialMetric");
  TestHelpers::db::test_compute_tag<
      gr::Tags::ShiftCompute<3, Frame::Inertial, DataVector>>("Shift");
  TestHelpers::db::test_compute_tag<
      gr::Tags::LapseCompute<3, Frame::Inertial, DataVector>>("Lapse");
  TestHelpers::db::test_compute_tag<
      gr::Tags::SqrtDetSpatialMetricCompute<3, Frame::Inertial, DataVector>>(
      "SqrtDetSpatialMetric");
  TestHelpers::db::test_compute_tag<gr::Tags::DetAndInverseSpatialMetricCompute<
      3, Frame::Inertial, DataVector>>(
      "Variables(DetSpatialMetric,InverseSpatialMetric)");
  TestHelpers::db::test_compute_tag<
      gr::Tags::DerivativesOfSpacetimeMetricCompute<3, Frame::Inertial>>(
      "DerivativesOfSpacetimeMetric");

  // Second, put the compute items into a data box and check that they
  // put the correct results
  // Let's start with a known spacetime metric, and then test the
  // compute items that depend on the spacetime metric
  DataVector used_for_size{5., 4.};
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-0.1, 0.1);

  const auto expected_spacetime_metric = [&generator, &distribution,
                                          &used_for_size]() {
    auto spacetime_metric_l =
        make_with_random_values<tnsr::aa<DataVector, 3, Frame::Inertial>>(
            make_not_null(&generator), make_not_null(&distribution),
            used_for_size);
    // Make sure spacetime_metric isn't singular.
    get<0, 0>(spacetime_metric_l) += -1.;
    for (size_t i = 1; i <= 3; ++i) {
      spacetime_metric_l.get(i, i) += 1.;
    }
    return spacetime_metric_l;
  }();

  const auto expected_spatial_metric =
      gr::spatial_metric(expected_spacetime_metric);
  const auto expected_det_and_inverse_spatial_metric =
      determinant_and_inverse(expected_spatial_metric);
  const auto expected_shift =
      gr::shift(expected_spacetime_metric,
                expected_det_and_inverse_spatial_metric.second);
  const auto expected_lapse =
      gr::lapse(expected_shift, expected_spacetime_metric);
  const auto expected_inverse_spacetime_metric = gr::inverse_spacetime_metric(
      expected_lapse, expected_shift,
      expected_det_and_inverse_spatial_metric.second);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>,
      db::AddComputeTags<
          gr::Tags::SpatialMetricCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::DetAndInverseSpatialMetricCompute<3, Frame::Inertial,
                                                      DataVector>,
          gr::Tags::SqrtDetSpatialMetricCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::ShiftCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::LapseCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::InverseSpacetimeMetricCompute<3, Frame::Inertial,
                                                  DataVector>,
          gr::Tags::SpacetimeNormalOneFormCompute<3, Frame::Inertial,
                                                  DataVector>,
          gr::Tags::SpacetimeNormalVectorCompute<3, Frame::Inertial,
                                                 DataVector>>>(
      expected_spacetime_metric);
  CHECK(db::get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(box) ==
        expected_spatial_metric);
  CHECK(db::get<gr::Tags::DetSpatialMetric<DataVector>>(box) ==
        expected_det_and_inverse_spatial_metric.first);
  CHECK(db::get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>(
            box) == expected_det_and_inverse_spatial_metric.second);
  CHECK(get(db::get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(box)) ==
        sqrt(get(expected_det_and_inverse_spatial_metric.first)));
  CHECK(db::get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(box) ==
        expected_shift);
  CHECK(db::get<gr::Tags::Lapse<DataVector>>(box) == expected_lapse);
  CHECK(
      db::get<gr::Tags::InverseSpacetimeMetric<3, Frame::Inertial, DataVector>>(
          box) == expected_inverse_spacetime_metric);
  CHECK(
      db::get<gr::Tags::SpacetimeNormalOneForm<3, Frame::Inertial, DataVector>>(
          box) ==
      gr::spacetime_normal_one_form<3, Frame::Inertial, DataVector>(
          expected_lapse));
  CHECK(
      db::get<gr::Tags::SpacetimeNormalVector<3, Frame::Inertial, DataVector>>(
          box) ==
      gr::spacetime_normal_vector<3, Frame::Inertial, DataVector>(
          expected_lapse, expected_shift));

  // Now let's put the lapse, shift, and spatial metric into the databox
  // and test that we can compute the correct spacetime metric
  const auto second_box = db::create<
      db::AddSimpleTags<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        gr::Tags::Lapse<DataVector>,
                        gr::Tags::Shift<3, Frame::Inertial, DataVector>>,
      db::AddComputeTags<
          gr::Tags::SpacetimeMetricCompute<3, Frame::Inertial, DataVector>>>(
      expected_spatial_metric, expected_lapse, expected_shift);
  CHECK_ITERABLE_APPROX(
      SINGLE_ARG(
          db::get<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>(
              second_box)),
      expected_spacetime_metric);

  // Now let's put the temporal and spatial derivatives of lapse, shift, and
  // spatial metric into the databox and test that we can assemple the
  // correct spatial and spacetime derivatives of the spacetime metric
  const auto deriv_spatial_metric =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const auto deriv_shift =
      make_with_random_values<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const auto deriv_lapse =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const auto dt_spatial_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const auto dt_shift =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  const auto dt_lapse = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  const auto expected_derivatives_of_spacetime_metric =
      gr::derivatives_of_spacetime_metric(
          expected_lapse, dt_lapse, deriv_lapse, expected_shift, dt_shift,
          deriv_shift, expected_spatial_metric, dt_spatial_metric,
          deriv_spatial_metric);

  const auto third_box = db::create<
      db::AddSimpleTags<
          gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
          gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<3, Frame::Inertial, DataVector>,
          ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>,
          ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>,
          ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>,
          ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>,
          ::Tags::dt<gr::Tags::Lapse<DataVector>>,
          ::Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataVector>>>,
      db::AddComputeTags<
          gr::Tags::DerivativesOfSpacetimeMetricCompute<3, Frame::Inertial>>>(
      expected_spatial_metric, expected_lapse, expected_shift,
      deriv_spatial_metric, deriv_lapse, deriv_shift, dt_spatial_metric,
      dt_lapse, dt_shift);
  CHECK(db::get<gr::Tags::DerivativesOfSpacetimeMetric<3, Frame::Inertial,
                                                       DataVector>>(
            third_box) == expected_derivatives_of_spacetime_metric);
}
