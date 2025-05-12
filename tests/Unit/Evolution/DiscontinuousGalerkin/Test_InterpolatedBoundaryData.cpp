// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Evolution/DiscontinuousGalerkin/InterpolatedBoundaryData.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GetOutput.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim>
void test() {
  const DataVector interpolated_data{3, 1.0};
  const Mesh<Dim - 1> mesh{3_st, Spectral::Basis::Legendre,
                     Spectral::Quadrature::Gauss};
  const std::vector<size_t> offsets{0_st, 2_st, 3_st};
  const InterpolatedBoundaryData<Dim> interpolated_boundary_data{
      {.data = interpolated_data, .target_mesh = mesh, .offsets = offsets}};
  CHECK(interpolated_boundary_data.boundary_data() == interpolated_data);
  CHECK(interpolated_boundary_data.target_mesh() == mesh);
  CHECK(interpolated_boundary_data.offsets() == offsets);
  CHECK(get_output(interpolated_boundary_data) ==
        std::string("boundary data = " + get_output(interpolated_data) +
                    "\ntarget mesh = " + get_output(mesh) +
                    "\noffsets = " + get_output(offsets)));
  const auto deserialized_interpolated_boundary_data =
      serialize_and_deserialize(interpolated_boundary_data);
  CHECK(interpolated_boundary_data == deserialized_interpolated_boundary_data);
  const DataVector interpolated_data_2{3, 1.5};
  const Mesh<Dim - 1> mesh_2{4_st, Spectral::Basis::Legendre,
                       Spectral::Quadrature::Gauss};
  const std::vector<size_t> offsets_2{2_st, 5_st, 7_st};
  const InterpolatedBoundaryData<Dim> interpolated_boundary_data_2{
      {.data = interpolated_data_2, .target_mesh = mesh, .offsets = offsets}};
  CHECK(interpolated_boundary_data != interpolated_boundary_data_2);
  const InterpolatedBoundaryData<Dim> interpolated_boundary_data_3{
      {.data = interpolated_data, .target_mesh = mesh_2, .offsets = offsets}};
  if constexpr (Dim > 1) {
    // All Mesh<0> are equivalent
    CHECK(interpolated_boundary_data != interpolated_boundary_data_3);
  }
  const InterpolatedBoundaryData<Dim> interpolated_boundary_data_4{
      {.data = interpolated_data, .target_mesh = mesh, .offsets = offsets_2}};
  CHECK(interpolated_boundary_data != interpolated_boundary_data_4);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.InterpolatedBoundaryData",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace evolution::dg
