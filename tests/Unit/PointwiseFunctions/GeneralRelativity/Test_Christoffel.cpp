// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "tests/Unit/DataStructures/TestHelpers.hpp"
#include "tests/Unit/PointwiseFunctions/GeneralRelativity/GrTestHelpers.hpp"
#include "tests/Unit/Pypp/Pypp.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestingFramework.hpp"

namespace {

template <size_t Dim, IndexType Index, typename DataType>
void test_christoffel(const DataType& used_for_size) {
  std::random_device r;
  const auto seed = r();
  std::mt19937 generator(seed);
  INFO("seed" << seed);
  std::uniform_real_distribution<> dist(-10., 10.);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  const auto d_metric =
      make_with_random_values<tnsr::abb<DataType, Dim, Frame::Inertial, Index>>(
          nn_generator, nn_dist, used_for_size);
  CHECK_ITERABLE_APPROX(
      (gr::christoffel_first_kind(d_metric)),
      (pypp::call<tnsr::abb<DataType, Dim, Frame::Inertial, Index>>(
          "GrTests", "christoffel_first_kind", d_metric)));
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
}
