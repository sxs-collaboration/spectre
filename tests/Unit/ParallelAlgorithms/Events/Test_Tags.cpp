// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Utilities/Literals.hpp"

namespace {
template <size_t Dim>
void test(const Mesh<Dim>& mesh) {
  const tnsr::I<DataVector, Dim, Frame::Grid> grid_coords{5_st, 7.2};
  const tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords{5_st, 7.2};
  const auto box = db::create<
      db::AddSimpleTags<domain::Tags::Mesh<Dim>,
                        domain::Tags::Coordinates<Dim, Frame::Grid>,
                        domain::Tags::Coordinates<Dim, Frame::Inertial>>,
      db::AddComputeTags<
          Events::Tags::ObserverMeshCompute<Dim>,
          Events::Tags::ObserverCoordinatesCompute<Dim, Frame::Grid>,
          Events::Tags::ObserverCoordinatesCompute<Dim, Frame::Inertial>>>(
      mesh, grid_coords, inertial_coords);
  CHECK(db::get<Events::Tags::ObserverMesh<Dim>>(box) == mesh);
  CHECK(db::get<Events::Tags::ObserverCoordinates<Dim, Frame::Grid>>(box) ==
        grid_coords);
  CHECK(db::get<Events::Tags::ObserverCoordinates<Dim, Frame::Inertial>>(box) ==
        inertial_coords);
  TestHelpers::db::test_simple_tag<Events::Tags::ObserverMesh<Dim>>(
      "ObserverMesh");
  TestHelpers::db::test_compute_tag<Events::Tags::ObserverMeshCompute<Dim>>(
      "ObserverMesh");
  TestHelpers::db::test_simple_tag<
      Events::Tags::ObserverCoordinates<Dim, Frame::Grid>>("GridCoordinates");
  TestHelpers::db::test_compute_tag<
      Events::Tags::ObserverCoordinatesCompute<Dim, Frame::Grid>>(
      "GridCoordinates");
  TestHelpers::db::test_simple_tag<
      Events::Tags::ObserverCoordinates<Dim, Frame::Inertial>>(
      "InertialCoordinates");
  TestHelpers::db::test_compute_tag<
      Events::Tags::ObserverCoordinatesCompute<Dim, Frame::Inertial>>(
      "InertialCoordinates");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Events.Tags", "[Unit]") {
  test(Mesh<1>{3, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto});
  test(Mesh<2>{3, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto});
  test(Mesh<3>{3, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto});
}
