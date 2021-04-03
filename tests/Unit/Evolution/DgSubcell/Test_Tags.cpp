// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

class DataVector;

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};

template <size_t Dim>
void test() {
  TestHelpers::db::test_simple_tag<evolution::dg::subcell::Tags::Mesh<Dim>>(
      "Subcell(Mesh)");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::NeighborDataForReconstructionAndRdmpTci<
          Dim>>("NeighborDataForReconstructionAndRdmpTci");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Logical>>(
      "LogicalCoordinates");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Grid>>(
      "GridCoordinates");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Inertial>>(
      "InertialCoordinates");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>>(
      "InverseJacobian(Logical,Grid)");

  TestHelpers::db::test_compute_tag<
      evolution::dg::subcell::Tags::LogicalCoordinatesCompute<Dim>>(
      "LogicalCoordinates");
  Mesh<Dim> subcell_mesh(5, Spectral::Basis::FiniteDifference,
                         Spectral::Quadrature::CellCentered);
  const auto logical_coords_box = db::create<
      db::AddSimpleTags<evolution::dg::subcell::Tags::Mesh<Dim>>,
      db::AddComputeTags<
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<Dim>>>(
      subcell_mesh);
  CHECK(db::get<evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Logical>>(
            logical_coords_box) == logical_coordinates(subcell_mesh));

  TestHelpers::db::test_compute_tag<
      evolution::dg::subcell::Tags::TciStatusCompute<Dim>>("TciStatus");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Tags",
                  "[Evolution][Unit]") {
  TestHelpers::db::test_simple_tag<evolution::dg::subcell::Tags::ActiveGrid>(
      "ActiveGrid");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToGrid>(
      "Det(InverseJacobian(Logical,Grid))");
  TestHelpers::db::test_simple_tag<evolution::dg::subcell::Tags::DidRollback>(
      "DidRollback");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::Inactive<Var1>>("Inactive(Var1)");
  TestHelpers::db::test_simple_tag<evolution::dg::subcell::Tags::Inactive<
      ::Tags::Variables<tmpl::list<Var1, Var2>>>>(
      "Inactive(Variables(Var1,Var2))");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::SubcellOptions>("SubcellOptions");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::TciGridHistory>("TciGridHistory");
  TestHelpers::db::test_simple_tag<evolution::dg::subcell::Tags::TciStatus>(
      "TciStatus");

  test<1>();
  test<2>();
  test<3>();
}
