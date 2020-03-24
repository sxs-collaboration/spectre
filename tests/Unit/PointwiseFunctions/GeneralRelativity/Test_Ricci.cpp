// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"

// IWYU pragma: no_include <boost/preprocessor/arithmetic/dec.hpp>
// IWYU pragma: no_include <boost/preprocessor/repetition/enum.hpp>
// IWYU pragma: no_include <boost/preprocessor/tuple/reverse.hpp>

namespace {
template <size_t Dim, IndexType TypeOfIndex, typename DataType>
void test_compute_item_in_databox(const DataType& used_for_size) noexcept {
  TestHelpers::db::test_compute_tag<
      gr::Tags::SpatialRicciCompute<3, Frame::Inertial, DataType>>(
      "SpatialRicci");

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

  const auto box = db::create<
      db::AddSimpleTags<gr::Tags::SpatialChristoffelSecondKind<
                            Dim, Frame::Inertial, DataType>,
                        ::Tags::deriv<gr::Tags::SpatialChristoffelSecondKind<
                                          Dim, Frame::Inertial, DataType>,
                                      tmpl::size_t<Dim>, Frame::Inertial>>,
      db::AddComputeTags<
          gr::Tags::SpatialRicciCompute<Dim, Frame::Inertial, DataType>>>(
      christoffel_2nd_kind, d_christoffel_2nd_kind);

  const auto expected =
      gr::ricci_tensor(christoffel_2nd_kind, d_christoffel_2nd_kind);
  CHECK_ITERABLE_APPROX(
      (db::get<gr::Tags::SpatialRicci<Dim, Frame::Inertial, DataType>>(box)),
      expected);
}

template <size_t Dim, IndexType TypeOfIndex, typename DataType>
void test_ricci(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::aa<DataType, Dim, Frame::Inertial, TypeOfIndex> (*)(
          const tnsr::Abb<DataType, Dim, Frame::Inertial, TypeOfIndex>&,
          const tnsr::aBcc<DataType, Dim, Frame::Inertial,
                           TypeOfIndex>&) noexcept>(
          &gr::ricci_tensor<Dim, Frame::Inertial, TypeOfIndex, DataType>),
      "Ricci", "ricci_tensor", {{{-1., 1.}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Ricci.",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_ricci, (1, 2, 3),
                                    (IndexType::Spatial, IndexType::Spacetime));
  test_compute_item_in_databox<3, IndexType::Spatial>(d);
  test_compute_item_in_databox<3, IndexType::Spatial>(dv);
}
