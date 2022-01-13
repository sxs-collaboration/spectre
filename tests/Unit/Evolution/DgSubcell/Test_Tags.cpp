// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcells.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

class DataVector;

namespace subcell = evolution::dg::subcell;

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};

template <size_t Dim>
void test() {
  TestHelpers::db::test_simple_tag<subcell::Tags::Mesh<Dim>>("Subcell(Mesh)");
  TestHelpers::db::test_simple_tag<
      subcell::Tags::NeighborDataForReconstruction<Dim>>(
      "NeighborDataForReconstruction");
  TestHelpers::db::test_simple_tag<subcell::Tags::DataForRdmpTci>(
      "DataForRdmpTci");
  TestHelpers::db::test_simple_tag<
      subcell::Tags::Coordinates<Dim, Frame::ElementLogical>>(
      "ElementLogicalCoordinates");
  TestHelpers::db::test_simple_tag<
      subcell::Tags::Coordinates<Dim, Frame::Grid>>("GridCoordinates");
  TestHelpers::db::test_simple_tag<
      subcell::Tags::Coordinates<Dim, Frame::Inertial>>("InertialCoordinates");
  TestHelpers::db::test_simple_tag<
      subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>>(
      "InverseJacobian(Logical,Grid)");
  TestHelpers::db::test_simple_tag<subcell::Tags::OnSubcellFaces<Var1, Dim>>(
      "OnSubcellFaces(Var1)");
  TestHelpers::db::test_simple_tag<subcell::Tags::OnSubcellFaces<
      ::Tags::Variables<tmpl::list<Var1, Var2>>, Dim>>(
      "OnSubcellFaces(Variables(Var1,Var2))");

  TestHelpers::db::test_compute_tag<
      subcell::Tags::LogicalCoordinatesCompute<Dim>>(
      "ElementLogicalCoordinates");
  TestHelpers::db::test_compute_tag<subcell::Tags::InertialCoordinatesCompute<
      ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                    Frame::Inertial>>>(
      "InertialCoordinates");
  Mesh<Dim> subcell_mesh(5, Spectral::Basis::FiniteDifference,
                         Spectral::Quadrature::CellCentered);
  const auto logical_coords_box = db::create<
      db::AddSimpleTags<subcell::Tags::Mesh<Dim>>,
      db::AddComputeTags<subcell::Tags::LogicalCoordinatesCompute<Dim>>>(
      subcell_mesh);
  CHECK(db::get<subcell::Tags::Coordinates<Dim, Frame::ElementLogical>>(
            logical_coords_box) == logical_coordinates(subcell_mesh));

  TestHelpers::db::test_compute_tag<subcell::Tags::TciStatusCompute<Dim>>(
      "TciStatus");

  auto active_coords_box = db::create<
      db::AddSimpleTags<domain::Tags::Coordinates<3, Frame::Inertial>,
                        subcell::Tags::Coordinates<3, Frame::Inertial>,
                        subcell::Tags::ActiveGrid>,
      db::AddComputeTags<
          subcell::Tags::ObserverCoordinatesCompute<3, Frame::Inertial>>>(
      tnsr::I<DataVector, 3, Frame::Inertial>{
          {{DataVector{8, 1.0}, DataVector{8, 3.0}, DataVector{8, 8.0}}}},
      tnsr::I<DataVector, 3, Frame::Inertial>{
          {{DataVector{27, 2.0}, DataVector{27, 5.0}, DataVector{27, 11.0}}}},
      subcell::ActiveGrid::Dg);
  CHECK(db::get<::Events::Tags::ObserverCoordinates<3, Frame::Inertial>>(
            active_coords_box) ==
        tnsr::I<DataVector, 3, Frame::Inertial>{
            {{DataVector{8, 1.0}, DataVector{8, 3.0}, DataVector{8, 8.0}}}});
  db::mutate<subcell::Tags::ActiveGrid>(
      make_not_null(&active_coords_box), [](const auto active_grid_ptr) {
        *active_grid_ptr = subcell::ActiveGrid::Subcell;
      });
  CHECK(
      db::get<::Events::Tags::ObserverCoordinates<3, Frame::Inertial>>(
          active_coords_box) ==
      tnsr::I<DataVector, 3, Frame::Inertial>{
          {{DataVector{27, 2.0}, DataVector{27, 5.0}, DataVector{27, 11.0}}}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Tags",
                  "[Evolution][Unit]") {
  TestHelpers::db::test_simple_tag<
      ::Events::Tags::ObserverCoordinates<1, Frame::Inertial>>(
      "InertialCoordinates");
  TestHelpers::db::test_simple_tag<
      ::Events::Tags::ObserverCoordinates<1, Frame::Grid>>("GridCoordinates");
  TestHelpers::db::test_simple_tag<subcell::Tags::ActiveGrid>("ActiveGrid");
  TestHelpers::db::test_simple_tag<
      subcell::fd::Tags::DetInverseJacobianLogicalToGrid>(
      "Det(InverseJacobian(Logical,Grid))");
  TestHelpers::db::test_simple_tag<subcell::Tags::DidRollback>("DidRollback");
  TestHelpers::db::test_simple_tag<subcell::Tags::Inactive<Var1>>(
      "Inactive(Var1)");
  TestHelpers::db::test_simple_tag<
      subcell::Tags::Inactive<::Tags::Variables<tmpl::list<Var1, Var2>>>>(
      "Variables(Inactive(Var1),Inactive(Var2))");
  TestHelpers::db::test_simple_tag<subcell::Tags::OnSubcells<Var1>>(
      "OnSubcells(Var1)");
  TestHelpers::db::test_simple_tag<
      subcell::Tags::OnSubcells<::Tags::Variables<tmpl::list<Var1, Var2>>>>(
      "Variables(OnSubcells(Var1),OnSubcells(Var2))");
  TestHelpers::db::test_simple_tag<subcell::Tags::SubcellOptions>(
      "SubcellOptions");
  TestHelpers::db::test_simple_tag<subcell::Tags::TciGridHistory>(
      "TciGridHistory");
  TestHelpers::db::test_simple_tag<subcell::Tags::TciStatus>("TciStatus");

  test<1>();
  test<2>();
  test<3>();
}
