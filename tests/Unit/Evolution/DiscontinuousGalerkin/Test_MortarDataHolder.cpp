// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim>
void test() {
  const Mesh<Dim> volume_mesh{4, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim - 1> face_mesh{4, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim - 1> mortar_mesh{5, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto};
  constexpr size_t number_of_components = 1 + Dim;
  DataVector local_mortar_data{
      mortar_mesh.number_of_grid_points() * number_of_components, 2.0};
  DataVector neighbor_mortar_data{
      mortar_mesh.number_of_grid_points() * number_of_components, 3.0};
  Scalar<DataVector> normal_magnitude{
      DataVector{face_mesh.number_of_grid_points(), 4.0}};
  Scalar<DataVector> det_j{DataVector{face_mesh.number_of_grid_points(), 5.0}};
  Scalar<DataVector> det_inv_j{
      DataVector{volume_mesh.number_of_grid_points(), 6.0}};
  MortarDataHolder<Dim> holder{};
  holder.local() = MortarData<Dim>{
      local_mortar_data, normal_magnitude, det_j,      det_inv_j,
      mortar_mesh,       face_mesh,        volume_mesh};
  holder.neighbor() = MortarData<Dim>{
      neighbor_mortar_data, std::nullopt, std::nullopt, std::nullopt,
      mortar_mesh,          std::nullopt, std::nullopt};

  std::string expected_output =
      MakeString{} << "Local mortar data:\n"
                   << "Mortar data: " << local_mortar_data << "\n"
                   << "Mortar mesh: " << mortar_mesh << "\n"
                   << "Face normal magnitude: " << normal_magnitude << "\n"
                   << "Face det(J): " << det_j << "\n"
                   << "Face mesh: " << face_mesh << "\n"
                   << "Volume det(invJ): " << det_inv_j << "\n"
                   << "Volume mesh: " << volume_mesh << "\n\n"
                   << "Neighbor mortar data:\n"
                   << "Mortar data: " << neighbor_mortar_data << "\n"
                   << "Mortar mesh: " << mortar_mesh << "\n"
                   << "Face normal magnitude: --\n"
                   << "Face det(J): --\n"
                   << "Face mesh: --\n"
                   << "Volume det(invJ): --\n"
                   << "Volume mesh: --\n\n";

  CHECK(get_output(holder) == expected_output);
  const auto deserialized_holder = serialize_and_deserialize(holder);
  CHECK(holder == deserialized_holder);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.MortarDataHolder", "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace evolution::dg
