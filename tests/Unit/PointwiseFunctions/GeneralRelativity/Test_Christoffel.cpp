// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template <typename Tag, typename Dim, typename Frame, typename>
struct deriv;
}  // namespace Tags
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace {
template <size_t Dim, IndexType Index, typename DataType>
void test_christoffel(const DataType& used_for_size) {
  tnsr::abb<DataType, Dim, Frame::Inertial, Index> (*f)(
      const tnsr::abb<DataType, Dim, Frame::Inertial, Index>&) =
      &gr::christoffel_first_kind<Dim, Frame::Inertial, Index, DataType>;
  pypp::check_with_random_values<1>(f, "Christoffel", "christoffel_first_kind",
                                    {{{-10., 10.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Christoffel",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");
  const DataVector dv(5);
  test_christoffel<1, IndexType::Spatial>(dv);
  test_christoffel<2, IndexType::Spatial>(dv);
  test_christoffel<3, IndexType::Spatial>(dv);
  test_christoffel<1, IndexType::Spacetime>(dv);
  test_christoffel<2, IndexType::Spacetime>(dv);
  test_christoffel<3, IndexType::Spacetime>(dv);
  test_christoffel<1, IndexType::Spatial>(0.);
  test_christoffel<2, IndexType::Spatial>(0.);
  test_christoffel<3, IndexType::Spatial>(0.);
  test_christoffel<1, IndexType::Spacetime>(0.);
  test_christoffel<2, IndexType::Spacetime>(0.);
  test_christoffel<3, IndexType::Spacetime>(0.);

  // Check that compute items work correctly in the DataBox
  // First, check that the names are correct
  TestHelpers::db::test_compute_tag<
      gr::Tags::SpatialChristoffelFirstKindCompute<3, Frame::Inertial,
                                                   DataVector>>(
      "SpatialChristoffelFirstKind");
  TestHelpers::db::test_compute_tag<
      gr::Tags::TraceSpatialChristoffelFirstKindCompute<3, Frame::Inertial,
                                                        DataVector>>(
      "TraceSpatialChristoffelFirstKind");
  TestHelpers::db::test_compute_tag<
      gr::Tags::SpatialChristoffelSecondKindCompute<3, Frame::Inertial,
                                                    DataVector>>(
      "SpatialChristoffelSecondKind");
  TestHelpers::db::test_compute_tag<
      gr::Tags::TraceSpatialChristoffelSecondKindCompute<3, Frame::Inertial,
                                                         DataVector>>(
      "TraceSpatialChristoffelSecondKind");
  TestHelpers::db::test_compute_tag<
      gr::Tags::SpacetimeChristoffelFirstKindCompute<3, Frame::Inertial,
                                                     DataVector>>(
      "SpacetimeChristoffelFirstKind");
  TestHelpers::db::test_compute_tag<
      gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<3, Frame::Inertial,
                                                          DataVector>>(
      "TraceSpacetimeChristoffelFirstKind");
  TestHelpers::db::test_compute_tag<
      gr::Tags::SpacetimeChristoffelSecondKindCompute<3, Frame::Inertial,
                                                      DataVector>>(
      "SpacetimeChristoffelSecondKind");

  // Check that the compute items return correct values
  const DataVector used_for_size{3., 4., 5.};
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-0.2, 0.2);

  const auto spacetime_metric = [&]() {
    auto spacetime_metric_l =
        make_with_random_values<tnsr::aa<DataVector, 3, Frame::Inertial>>(
            make_not_null(&generator), make_not_null(&distribution),
            used_for_size);
    // Make sure spacetime_metric isn't singular
    get<0, 0>(spacetime_metric_l) += -1.;
    for (size_t i = 1; i <= 3; ++i) {
      spacetime_metric_l.get(i, i) += 1.;
    }
    return spacetime_metric_l;
  }();

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

  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto det_and_inverse_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& inverse_spatial_metric = det_and_inverse_spatial_metric.second;
  const auto shift = gr::shift(spacetime_metric, inverse_spatial_metric);
  const auto lapse = gr::lapse(shift, spacetime_metric);
  const auto inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);

  const auto derivatives_of_spacetime_metric =
      gr::derivatives_of_spacetime_metric(
          lapse, dt_lapse, deriv_lapse, shift, dt_shift, deriv_shift,
          spatial_metric, dt_spatial_metric, deriv_spatial_metric);

  const auto expected_spatial_christoffel_first_kind =
      gr::christoffel_first_kind(deriv_spatial_metric);
  const auto expected_trace_spatial_christoffel_first_kind = trace_last_indices(
      expected_spatial_christoffel_first_kind, inverse_spatial_metric);
  const auto expected_spatial_christoffel_second_kind =
      raise_or_lower_first_index(expected_spatial_christoffel_first_kind,
                                 inverse_spatial_metric);
  const auto expected_trace_spatial_christoffel_second_kind =
      trace_last_indices(expected_spatial_christoffel_second_kind,
                         inverse_spatial_metric);

  const auto expected_spacetime_christoffel_first_kind =
      gr::christoffel_first_kind(derivatives_of_spacetime_metric);
  const auto expected_trace_spacetime_christoffel_first_kind =
      trace_last_indices(expected_spacetime_christoffel_first_kind,
                         inverse_spacetime_metric);
  const auto expected_spacetime_christoffel_second_kind =
      raise_or_lower_first_index(expected_spacetime_christoffel_first_kind,
                                 inverse_spacetime_metric);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
          ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>,
          gr::Tags::InverseSpacetimeMetric<3, Frame::Inertial, DataVector>,
          gr::Tags::DerivativesOfSpacetimeMetric<3, Frame::Inertial,
                                                 DataVector>>,
      db::AddComputeTags<gr::Tags::SpatialChristoffelFirstKindCompute<
                             3, Frame::Inertial, DataVector>,
                         gr::Tags::TraceSpatialChristoffelFirstKindCompute<
                             3, Frame::Inertial, DataVector>,
                         gr::Tags::SpatialChristoffelSecondKindCompute<
                             3, Frame::Inertial, DataVector>,
                         gr::Tags::TraceSpatialChristoffelSecondKindCompute<
                             3, Frame::Inertial, DataVector>,
                         gr::Tags::SpacetimeChristoffelFirstKindCompute<
                             3, Frame::Inertial, DataVector>,
                         gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<
                             3, Frame::Inertial, DataVector>,
                         gr::Tags::SpacetimeChristoffelSecondKindCompute<
                             3, Frame::Inertial, DataVector>>>(
      inverse_spatial_metric, deriv_spatial_metric, inverse_spacetime_metric,
      derivatives_of_spacetime_metric);

  CHECK(db::get<gr::Tags::SpatialChristoffelFirstKind<3, Frame::Inertial,
                                                      DataVector>>(box) ==
        expected_spatial_christoffel_first_kind);
  CHECK(db::get<gr::Tags::TraceSpatialChristoffelFirstKind<3, Frame::Inertial,
                                                           DataVector>>(box) ==
        expected_trace_spatial_christoffel_first_kind);
  CHECK(db::get<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial,
                                                       DataVector>>(box) ==
        expected_spatial_christoffel_second_kind);
  CHECK(db::get<gr::Tags::TraceSpatialChristoffelSecondKind<3, Frame::Inertial,
                                                            DataVector>>(box) ==
        expected_trace_spatial_christoffel_second_kind);

  CHECK(db::get<gr::Tags::SpacetimeChristoffelFirstKind<3, Frame::Inertial,
                                                        DataVector>>(box) ==
        expected_spacetime_christoffel_first_kind);
  CHECK(db::get<gr::Tags::TraceSpacetimeChristoffelFirstKind<3, Frame::Inertial,
                                                             DataVector>>(
            box) == expected_trace_spacetime_christoffel_first_kind);
  CHECK(db::get<gr::Tags::SpacetimeChristoffelSecondKind<3, Frame::Inertial,
                                                         DataVector>>(box) ==
        expected_spacetime_christoffel_second_kind);
}
