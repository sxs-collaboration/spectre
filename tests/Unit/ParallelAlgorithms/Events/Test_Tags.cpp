// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>

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
  const InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      inv_jac{5_st, 9.3};
  const Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial> jac{
      5_st, 10.3};
  const Scalar<DataVector> det_inv_jac{5_st, 11.3};
  const tnsr::I<DataVector, Dim, Frame::Inertial> mesh_velocity{5_st, 8.3};
  const auto box = db::create<
      db::AddSimpleTags<
          domain::Tags::Mesh<Dim>, domain::Tags::Coordinates<Dim, Frame::Grid>,
          domain::Tags::Coordinates<Dim, Frame::Inertial>,
          domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                        Frame::Inertial>,
          domain::Tags::Jacobian<Dim, Frame::ElementLogical, Frame::Inertial>,
          domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
          domain::Tags::MeshVelocity<Dim, Frame::Inertial>>,
      db::AddComputeTags<
          ::Events::Tags::ObserverMeshCompute<Dim>,
          Events::Tags::ObserverCoordinatesCompute<Dim, Frame::Grid>,
          Events::Tags::ObserverCoordinatesCompute<Dim, Frame::Inertial>,
          Events::Tags::ObserverInverseJacobianCompute<
              Dim, Frame::ElementLogical, Frame::Inertial>,
          Events::Tags::ObserverJacobianCompute<Dim, Frame::ElementLogical,
                                                Frame::Inertial>,
          Events::Tags::ObserverDetInvJacobianCompute<Frame::ElementLogical,
                                                      Frame::Inertial>,
          Events::Tags::ObserverMeshVelocityCompute<Dim, Frame::Inertial>>>(
      mesh, grid_coords, inertial_coords, inv_jac, jac, det_inv_jac,
      std::optional{mesh_velocity});
  CHECK(db::get<Events::Tags::ObserverMesh<Dim>>(box) == mesh);
  CHECK(db::get<Events::Tags::ObserverCoordinates<Dim, Frame::Grid>>(box) ==
        grid_coords);
  CHECK(db::get<Events::Tags::ObserverCoordinates<Dim, Frame::Inertial>>(box) ==
        inertial_coords);
  CHECK(
      db::get<Events::Tags::ObserverInverseJacobian<Dim, Frame::ElementLogical,
                                                    Frame::Inertial>>(box) ==
      inv_jac);
  CHECK(db::get<Events::Tags::ObserverJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>>(box) == jac);
  CHECK(db::get<Events::Tags::ObserverDetInvJacobian<Frame::ElementLogical,
                                                     Frame::Inertial>>(box) ==
        det_inv_jac);
  CHECK(db::get<Events::Tags::ObserverMeshVelocity<Dim, Frame::Inertial>>(
            box) == mesh_velocity);
  TestHelpers::db::test_simple_tag<Events::Tags::ObserverMesh<Dim>>(
      "ObserverMesh");
  TestHelpers::db::test_compute_tag<::Events::Tags::ObserverMeshCompute<Dim>>(
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
  TestHelpers::db::test_simple_tag<Events::Tags::ObserverInverseJacobian<
      Dim, Frame::ElementLogical, Frame::Inertial>>(
      "InverseJacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_compute_tag<
      Events::Tags::ObserverInverseJacobianCompute<Dim, Frame::ElementLogical,
                                                   Frame::Inertial>>(
      "InverseJacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_simple_tag<Events::Tags::ObserverJacobian<
      Dim, Frame::ElementLogical, Frame::Inertial>>(
      "Jacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_compute_tag<Events::Tags::ObserverJacobianCompute<
      Dim, Frame::ElementLogical, Frame::Inertial>>(
      "Jacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_simple_tag<Events::Tags::ObserverDetInvJacobian<
      Frame::ElementLogical, Frame::Inertial>>(
      "DetInvJacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_compute_tag<Events::Tags::ObserverDetInvJacobianCompute<
      Frame::ElementLogical, Frame::Inertial>>(
      "DetInvJacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_simple_tag<
      Events::Tags::ObserverMeshVelocity<Dim, Frame::Inertial>>(
      "InertialMeshVelocity");
  TestHelpers::db::test_compute_tag<
      Events::Tags::ObserverMeshVelocityCompute<Dim, Frame::Inertial>>(
      "InertialMeshVelocity");
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
