// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/H5/TensorData.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
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

  const ElementVolumeData evd0(
      "Element0", {tc0, tc1, tc2}, {2, 2},
      {SpatialDiscretization::Basis::Legendre,
       SpatialDiscretization::Basis::Chebyshev},
      {SpatialDiscretization::Quadrature::Gauss,
       SpatialDiscretization::Quadrature::GaussLobatto});
  const auto after_evd0 = serialize_and_deserialize(evd0);
  CHECK(after_evd0.extents == evd0.extents);
  CHECK(after_evd0.tensor_components == evd0.tensor_components);
  CHECK(after_evd0.basis == evd0.basis);
  CHECK(after_evd0.quadrature == evd0.quadrature);
  CHECK(after_evd0.element_name == evd0.element_name);
  CHECK(after_evd0 == evd0);
  // Check operator==
  CHECK(evd0 !=
        ElementVolumeData("Element0", {tc0, tc1, tc2}, {3, 2},
                          {SpatialDiscretization::Basis::Legendre,
                           SpatialDiscretization::Basis::Chebyshev},
                          {SpatialDiscretization::Quadrature::Gauss,
                           SpatialDiscretization::Quadrature::GaussLobatto}));
  CHECK(evd0 !=
        ElementVolumeData("Element0", {tc2, tc1, tc0}, {2, 2},
                          {SpatialDiscretization::Basis::Legendre,
                           SpatialDiscretization::Basis::Chebyshev},
                          {SpatialDiscretization::Quadrature::Gauss,
                           SpatialDiscretization::Quadrature::GaussLobatto}));
  CHECK(evd0 !=
        ElementVolumeData("Element0", {tc0, tc1, tc2}, {2, 2},
                          {SpatialDiscretization::Basis::Chebyshev,
                           SpatialDiscretization::Basis::Chebyshev},
                          {SpatialDiscretization::Quadrature::Gauss,
                           SpatialDiscretization::Quadrature::GaussLobatto}));
  CHECK(evd0 !=
        ElementVolumeData("Element0", {tc0, tc1, tc2}, {2, 2},
                          {SpatialDiscretization::Basis::Legendre,
                           SpatialDiscretization::Basis::Chebyshev},
                          {SpatialDiscretization::Quadrature::GaussLobatto,
                           SpatialDiscretization::Quadrature::GaussLobatto}));
  CHECK(evd0 !=
        ElementVolumeData("Element1", {tc0, tc1, tc2}, {2, 2},
                          {SpatialDiscretization::Basis::Legendre,
                           SpatialDiscretization::Basis::Chebyshev},
                          {SpatialDiscretization::Quadrature::Gauss,
                           SpatialDiscretization::Quadrature::GaussLobatto}));
  CHECK(evd0 ==
        ElementVolumeData("Element0", {tc0, tc1, tc2}, {2, 2},
                          {SpatialDiscretization::Basis::Legendre,
                           SpatialDiscretization::Basis::Chebyshev},
                          {SpatialDiscretization::Quadrature::Gauss,
                           SpatialDiscretization::Quadrature::GaussLobatto}));
  CHECK(ElementVolumeData(
            ElementId<1>{0, {{{1, 1}}}}, {tc0, tc1, tc2},
            Mesh<1>{3, SpatialDiscretization::Basis::Legendre,
                    SpatialDiscretization::Quadrature::GaussLobatto}) ==
        ElementVolumeData("[B0,(L1I1)]", {tc0, tc1, tc2}, {3},
                          {SpatialDiscretization::Basis::Legendre},
                          {SpatialDiscretization::Quadrature::GaussLobatto}));
  CHECK(ElementVolumeData(
            ElementId<2>{0, {{{1, 1}, {2, 0}}}}, {tc0, tc1, tc2},
            Mesh<2>{{{3, 4}},
                    {{SpatialDiscretization::Basis::Legendre,
                      SpatialDiscretization::Basis::Chebyshev}},
                    {{SpatialDiscretization::Quadrature::Gauss,
                      SpatialDiscretization::Quadrature::GaussLobatto}}}) ==
        ElementVolumeData("[B0,(L1I1,L2I0)]", {tc0, tc1, tc2}, {3, 4},
                          {SpatialDiscretization::Basis::Legendre,
                           SpatialDiscretization::Basis::Chebyshev},
                          {SpatialDiscretization::Quadrature::Gauss,
                           SpatialDiscretization::Quadrature::GaussLobatto}));
  CHECK(ElementVolumeData(
            ElementId<3>{0, {{{1, 1}, {2, 0}, {2, 1}}}}, {tc0, tc1, tc2},
            Mesh<3>{{{3, 4, 5}},
                    {{SpatialDiscretization::Basis::Legendre,
                      SpatialDiscretization::Basis::Chebyshev,
                      SpatialDiscretization::Basis::Legendre}},
                    {{SpatialDiscretization::Quadrature::Gauss,
                      SpatialDiscretization::Quadrature::GaussLobatto,
                      SpatialDiscretization::Quadrature::Gauss}}}) ==
        ElementVolumeData("[B0,(L1I1,L2I0,L2I1)]", {tc0, tc1, tc2}, {3, 4, 5},
                          {SpatialDiscretization::Basis::Legendre,
                           SpatialDiscretization::Basis::Chebyshev,
                           SpatialDiscretization::Basis::Legendre},
                          {SpatialDiscretization::Quadrature::Gauss,
                           SpatialDiscretization::Quadrature::GaussLobatto,
                           SpatialDiscretization::Quadrature::Gauss}));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.TensorData", "[Unit]") {
  test<DataVector>();
  test<std::vector<float>>();
}
