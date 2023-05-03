// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

// IWYU pragma: no_include <boost/preprocessor/arithmetic/dec.hpp>
// IWYU pragma: no_include <boost/preprocessor/repetition/enum.hpp>
// IWYU pragma: no_include <boost/preprocessor/tuple/reverse.hpp>

namespace {
template <size_t Dim, IndexType TypeOfIndex, typename DataType>
void test_compute_item_in_databox(const DataType& used_for_size) {
  TestHelpers::db::test_compute_tag<
      gr::Tags::SpatialRicciCompute<DataType, Dim, Frame::Inertial>>(
      "SpatialRicci");
  TestHelpers::db::test_compute_tag<
      gr::Tags::SpatialRicciScalarCompute<DataType, Dim, Frame::Inertial>>(
      "SpatialRicciScalar");

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-3.0, 3.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto christoffel_2nd_kind = make_with_random_values<
      tnsr::Abb<DataType, Dim, Frame::Inertial, TypeOfIndex>>(
      nn_generator, nn_distribution, used_for_size);
  const auto d_christoffel_2nd_kind = make_with_random_values<
      tnsr::aBcc<DataType, Dim, Frame::Inertial, TypeOfIndex>>(
      nn_generator, nn_distribution, used_for_size);
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<Dim, DataType, Frame::Inertial>(
          nn_generator, used_for_size);
  const auto inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::SpatialChristoffelSecondKind<DataType, Dim>,
          ::Tags::deriv<gr::Tags::SpatialChristoffelSecondKind<DataType, Dim>,
                        tmpl::size_t<Dim>, Frame::Inertial>,
          gr::Tags::InverseSpatialMetric<DataType, Dim>>,
      db::AddComputeTags<
          gr::Tags::SpatialRicciCompute<DataType, Dim, Frame::Inertial>,
          gr::Tags::SpatialRicciScalarCompute<DataType, Dim, Frame::Inertial>>>(
      christoffel_2nd_kind, d_christoffel_2nd_kind, inv_spatial_metric);

  const auto expected =
      gr::ricci_tensor(christoffel_2nd_kind, d_christoffel_2nd_kind);
  const auto expected_spatial_scalar =
      gr::ricci_scalar(expected, inv_spatial_metric);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::SpatialRicci<DataType, Dim>>(box)),
                        expected);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::SpatialRicciScalar<DataType>>(box)),
                        expected_spatial_scalar);
}

template <size_t Dim, IndexType TypeOfIndex, typename DataType>
void test_ricci(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::aa<DataType, Dim, Frame::Inertial, TypeOfIndex> (*)(
          const tnsr::Abb<DataType, Dim, Frame::Inertial, TypeOfIndex>&,
          const tnsr::aBcc<DataType, Dim, Frame::Inertial, TypeOfIndex>&)>(
          &gr::ricci_tensor<Dim, Frame::Inertial, TypeOfIndex, DataType>),
      "Ricci", "ricci_tensor", {{{-1., 1.}}}, used_for_size);
}

template <size_t Dim, IndexType TypeOfIndex, typename DataType>
void test_ricci_scalar(const DataType& used_for_size) {
  Scalar<DataType> (*f)(
      const tnsr::aa<DataType, Dim, Frame::Inertial, TypeOfIndex>&,
      const tnsr::AA<DataType, Dim, Frame::Inertial, TypeOfIndex>&) =
      &gr::ricci_scalar<Dim, Frame::Inertial, TypeOfIndex, DataType>;
  pypp::check_with_random_values<1>(f, "RicciScalar", "ricci_scalar",
                                    {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Ricci.",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_ricci, (1, 2, 3),
                                    (IndexType::Spatial, IndexType::Spacetime));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_ricci_scalar, (1, 2, 3),
                                    (IndexType::Spatial, IndexType::Spacetime));
  test_compute_item_in_databox<3, IndexType::Spatial>(d);
  test_compute_item_in_databox<3, IndexType::Spatial>(dv);
}
