// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

namespace {
template <typename DataType>
void test() {
  TensorComponent tc0("T_x", DataType{8.9, 7.6, -3.4, 9.0});
  TensorComponent tc1("T_y", DataType{8.9, 7.6, -3.4, 9.0});
  TensorComponent tc2("T_x", DataType{8.9, 7.6, -3.4, 9.1});
  CHECK(get_output(tc0) == "(T_x, (8.9,7.6,-3.4,9))");
  CHECK(tc0 == tc0);
  CHECK(tc0 != tc1);
  CHECK(tc0 != tc2);
  CHECK(tc1 != tc2);
  test_serialization(tc0);

  const ExtentsAndTensorVolumeData etvd0({2, 2}, {tc0, tc1, tc2});
  const auto after = serialize_and_deserialize(etvd0);
  CHECK(after.extents == etvd0.extents);
  CHECK(after.tensor_components == etvd0.tensor_components);
  CHECK(after == etvd0);
  // Check operator==
  CHECK(etvd0 != ExtentsAndTensorVolumeData({3, 2}, {tc0, tc1, tc2}));
  CHECK(etvd0 != ExtentsAndTensorVolumeData({2, 2}, {tc2, tc1, tc0}));
  CHECK(etvd0 == ExtentsAndTensorVolumeData({2, 2}, {tc0, tc1, tc2}));

  const ElementVolumeData evd0(
      {2, 2}, {tc0, tc1, tc2},
      {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev},
      {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto},
      "Element0");
  const auto after_evd0 = serialize_and_deserialize(evd0);
  CHECK(after_evd0.extents == evd0.extents);
  CHECK(after_evd0.tensor_components == evd0.tensor_components);
  CHECK(after_evd0.basis == evd0.basis);
  CHECK(after_evd0.quadrature == evd0.quadrature);
  CHECK(after_evd0.element_name == evd0.element_name);
  CHECK(after_evd0 == evd0);
  // Check operator==
  CHECK(evd0 !=
        ElementVolumeData(
            {3, 2}, {tc0, tc1, tc2},
            {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev},
            {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto},
            "Element0"));
  CHECK(evd0 !=
        ElementVolumeData(
            {2, 2}, {tc2, tc1, tc0},
            {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev},
            {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto},
            "Element0"));
  CHECK(evd0 !=
        ElementVolumeData(
            {2, 2}, {tc0, tc1, tc2},
            {Spectral::Basis::Chebyshev, Spectral::Basis::Chebyshev},
            {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto},
            "Element0"));
  CHECK(evd0 != ElementVolumeData(
                    {2, 2}, {tc0, tc1, tc2},
                    {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev},
                    {Spectral::Quadrature::GaussLobatto,
                     Spectral::Quadrature::GaussLobatto},
                    "Element0"));
  CHECK(evd0 !=
        ElementVolumeData(
            {2, 2}, {tc0, tc1, tc2},
            {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev},
            {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto},
            "Element1"));
  CHECK(evd0 ==
        ElementVolumeData(
            {2, 2}, {tc0, tc1, tc2},
            {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev},
            {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto},
            "Element0"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.TensorData", "[Unit]") {
  test<DataVector>();
  test<std::vector<float>>();
}
