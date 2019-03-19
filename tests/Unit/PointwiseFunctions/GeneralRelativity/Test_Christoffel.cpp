// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

/// \cond
namespace GeneralizedHarmonic {
namespace Tags {
template <size_t SpatialDim, typename Frame>
struct DerivativesOfSpacetimeMetricCompute;
}  // namespace Tags
}  // namespace GeneralizedHarmonic
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

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Christoffel.",
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

  // Check that the compute items return correct values
  const DataVector test_vector{5., 4.};
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(test_vector,
                                                                0.);
  get<0, 0>(spatial_metric) = 1.5;
  get<0, 1>(spatial_metric) = 0.1;
  get<0, 2>(spatial_metric) = 0.2;
  get<1, 1>(spatial_metric) = 1.4;
  get<1, 2>(spatial_metric) = 0.2;
  get<2, 2>(spatial_metric) = 1.3;

  auto lapse = make_with_value<Scalar<DataVector>>(test_vector, 0.94);

  auto shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(test_vector, 0.);
  get<0>(shift) = 0.5;
  get<1>(shift) = 0.6;
  get<2>(shift) = 0.7;

  auto dt_spatial_metric =
      make_with_value<tnsr::ii<DataVector, 3, Frame::Inertial>>(test_vector,
                                                                0.);
  get<0, 0>(dt_spatial_metric) = 0.05;
  get<0, 1>(dt_spatial_metric) = 0.01;
  get<0, 2>(dt_spatial_metric) = 0.02;
  get<1, 1>(dt_spatial_metric) = 0.04;
  get<1, 2>(dt_spatial_metric) = 0.02;
  get<2, 2>(dt_spatial_metric) = 0.03;

  auto dt_lapse = make_with_value<Scalar<DataVector>>(test_vector, 0.);
  get(dt_lapse) = 0.04;

  auto dt_shift =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(test_vector, 0.);
  get<0>(dt_shift) = 0.05;
  get<1>(dt_shift) = 0.06;
  get<2>(dt_shift) = 0.07;

  auto deriv_spatial_metric =
      make_with_value<tnsr::ijj<DataVector, 3, Frame::Inertial>>(test_vector,
                                                                 0.);
  get<0, 0, 0>(deriv_spatial_metric) = 0.1;
  get<0, 0, 1>(deriv_spatial_metric) = 0.2;
  get<0, 0, 2>(deriv_spatial_metric) = 0.3;
  get<0, 1, 1>(deriv_spatial_metric) = 0.2;
  get<0, 1, 2>(deriv_spatial_metric) = 0.1;
  get<0, 2, 2>(deriv_spatial_metric) = 0.2;
  get<1, 0, 0>(deriv_spatial_metric) = 0.3;
  get<1, 0, 1>(deriv_spatial_metric) = 0.2;
  get<1, 0, 2>(deriv_spatial_metric) = 0.1;
  get<1, 1, 1>(deriv_spatial_metric) = 0.2;
  get<1, 1, 2>(deriv_spatial_metric) = 0.3;
  get<1, 2, 2>(deriv_spatial_metric) = 0.2;
  get<2, 0, 0>(deriv_spatial_metric) = -0.1;
  get<2, 0, 1>(deriv_spatial_metric) = -0.2;
  get<2, 0, 2>(deriv_spatial_metric) = -0.3;
  get<2, 1, 1>(deriv_spatial_metric) = -0.2;
  get<2, 1, 2>(deriv_spatial_metric) = -0.1;
  get<2, 2, 2>(deriv_spatial_metric) = -0.2;

  auto deriv_lapse =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(test_vector, 0.);
  get<0>(deriv_lapse) = -0.1;
  get<1>(deriv_lapse) = -0.2;
  get<2>(deriv_lapse) = -0.3;

  auto deriv_shift = make_with_value<tnsr::iJ<DataVector, 3, Frame::Inertial>>(
      test_vector, 0.);
  get<0, 0>(deriv_shift) = -0.05;
  get<0, 1>(deriv_shift) = 0.06;
  get<0, 2>(deriv_shift) = 0.07;
  get<1, 0>(deriv_shift) = -0.03;
  get<1, 1>(deriv_shift) = 0.04;
  get<1, 2>(deriv_shift) = 0.03;
  get<2, 0>(deriv_shift) = 0.01;
  get<2, 1>(deriv_shift) = 0.02;
  get<2, 2>(deriv_shift) = -0.01;

  const auto& derivatives_of_spacetime_metric =
      gr::derivatives_of_spacetime_metric(
          lapse, dt_lapse, deriv_lapse, shift, dt_shift, deriv_shift,
          spatial_metric, dt_spatial_metric, deriv_spatial_metric);

  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto& inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);

  const auto& spacetime_christoffel_first_kind =
      gr::christoffel_first_kind(derivatives_of_spacetime_metric);
  const auto& trace_spacetime_christoffel_first_kind = trace_last_indices(
      spacetime_christoffel_first_kind, inverse_spacetime_metric);
  const auto& spatial_christoffel_first_kind =
      gr::christoffel_first_kind(deriv_spatial_metric);
  const auto& trace_spatial_christoffel_first_kind = trace_last_indices(
      spatial_christoffel_first_kind, inverse_spatial_metric);
  const auto& spacetime_christoffel_second_kind = raise_or_lower_first_index(
      spacetime_christoffel_first_kind, inverse_spacetime_metric);
  const auto& spatial_christoffel_second_kind = raise_or_lower_first_index(
      spatial_christoffel_first_kind, inverse_spatial_metric);

  const auto box = db::create<
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
          gr::Tags::SpacetimeMetricCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::DetAndInverseSpatialMetricCompute<3, Frame::Inertial,
                                                      DataVector>,
          gr::Tags::SqrtDetSpatialMetricCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::InverseSpatialMetricCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::InverseSpacetimeMetricCompute<3, Frame::Inertial,
                                                  DataVector>,
          GeneralizedHarmonic::Tags::DerivativesOfSpacetimeMetricCompute<
              3, Frame::Inertial>,
          gr::Tags::SpacetimeChristoffelFirstKindCompute<3, Frame::Inertial,
                                                         DataVector>,
          gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<
              3, Frame::Inertial, DataVector>,
          gr::Tags::SpatialChristoffelFirstKindCompute<3, Frame::Inertial,
                                                       DataVector>,
          gr::Tags::TraceSpatialChristoffelFirstKindCompute<3, Frame::Inertial,
                                                            DataVector>,
          gr::Tags::SpacetimeChristoffelSecondKindCompute<3, Frame::Inertial,
                                                          DataVector>,
          gr::Tags::SpatialChristoffelSecondKindCompute<3, Frame::Inertial,
                                                        DataVector>  //,

          >>(spatial_metric, lapse, shift, deriv_spatial_metric, deriv_lapse,
             deriv_shift, dt_spatial_metric, dt_lapse, dt_shift);

  CHECK(db::get<gr::Tags::SpacetimeChristoffelFirstKind<3, Frame::Inertial,
                                                        DataVector>>(box) ==
        spacetime_christoffel_first_kind);
  CHECK(db::get<gr::Tags::TraceSpacetimeChristoffelFirstKind<3, Frame::Inertial,
                                                             DataVector>>(
            box) == trace_spacetime_christoffel_first_kind);
  CHECK(db::get<gr::Tags::SpatialChristoffelFirstKind<3, Frame::Inertial,
                                                      DataVector>>(box) ==
        spatial_christoffel_first_kind);
  CHECK(db::get<gr::Tags::TraceSpatialChristoffelFirstKind<3, Frame::Inertial,
                                                           DataVector>>(box) ==
        trace_spatial_christoffel_first_kind);
  CHECK(db::get<gr::Tags::SpacetimeChristoffelSecondKind<3, Frame::Inertial,
                                                         DataVector>>(box) ==
        spacetime_christoffel_second_kind);
  CHECK(db::get<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial,
                                                       DataVector>>(box) ==
        spatial_christoffel_second_kind);
}
