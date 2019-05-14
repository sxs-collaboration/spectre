// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

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
  CHECK(gr::Tags::SpatialChristoffelFirstKind<3, Frame::Inertial,
                                              DataVector>::name() ==
        "SpatialChristoffelFirstKind");
  CHECK(gr::Tags::TraceSpatialChristoffelFirstKind<3, Frame::Inertial,
                                                   DataVector>::name() ==
        "TraceSpatialChristoffelFirstKind");
  CHECK(gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial,
                                               DataVector>::name() ==
        "SpatialChristoffelSecondKind");
  CHECK(gr::Tags::TraceSpatialChristoffelSecondKind<3, Frame::Inertial,
                                                    DataVector>::name() ==
        "TraceSpatialChristoffelSecondKind");

  CHECK(gr::Tags::SpatialChristoffelFirstKindCompute<3, Frame::Inertial,
                                                     DataVector>::name() ==
        "SpatialChristoffelFirstKind");
  CHECK(gr::Tags::TraceSpatialChristoffelFirstKindCompute<3, Frame::Inertial,
                                                          DataVector>::name() ==
        "TraceSpatialChristoffelFirstKind");
  CHECK(gr::Tags::SpatialChristoffelSecondKindCompute<3, Frame::Inertial,
                                                      DataVector>::name() ==
        "SpatialChristoffelSecondKind");

  // Check that the compute items return correct values
  const DataVector used_for_size{3., 4., 5.};
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-0.2, 0.2);

  auto spacetime_metric =
      make_with_random_values<tnsr::aa<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  get<0, 0>(spacetime_metric) += -1.;
  for (size_t i = 1; i <= 3; ++i) {
    spacetime_metric.get(i, i) += 1.;
  }

  const auto deriv_spatial_metric =
      make_with_random_values<tnsr::ijj<DataVector, 3, Frame::Inertial>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto det_and_inverse_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& inverse_spatial_metric = det_and_inverse_spatial_metric.second;

  const auto spatial_christoffel_first_kind =
      gr::christoffel_first_kind(deriv_spatial_metric);
  const auto trace_spatial_christoffel_first_kind = trace_last_indices(
      spatial_christoffel_first_kind, inverse_spatial_metric);
  const auto spatial_christoffel_second_kind = raise_or_lower_first_index(
      spatial_christoffel_first_kind, inverse_spatial_metric);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
          ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>,
      db::AddComputeTags<gr::Tags::SpatialChristoffelFirstKindCompute<
                             3, Frame::Inertial, DataVector>,
                         gr::Tags::TraceSpatialChristoffelFirstKindCompute<
                             3, Frame::Inertial, DataVector>,
                         gr::Tags::SpatialChristoffelSecondKindCompute<
                             3, Frame::Inertial, DataVector>>>(
      inverse_spatial_metric, deriv_spatial_metric);

  CHECK(db::get<gr::Tags::SpatialChristoffelFirstKind<3, Frame::Inertial,
                                                      DataVector>>(box) ==
        spatial_christoffel_first_kind);
  CHECK(db::get<gr::Tags::TraceSpatialChristoffelFirstKind<3, Frame::Inertial,
                                                           DataVector>>(box) ==
        trace_spatial_christoffel_first_kind);
  CHECK(db::get<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial,
                                                       DataVector>>(box) ==
        spatial_christoffel_second_kind);
}
