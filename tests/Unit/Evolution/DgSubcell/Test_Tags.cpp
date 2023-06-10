// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/MethodOrder.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/ObserverMesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcells.hpp"
#include "Evolution/DgSubcell/Tags/ReconstructionOrder.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/FiniteDifference/DerivativeOrder.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/CloneUniquePtrs.hpp"

namespace subcell = evolution::dg::subcell;

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};

template <size_t Dim>
void test(const bool moving_mesh) {
  TestHelpers::db::test_simple_tag<subcell::Tags::SubcellOptions<Dim>>(
      "SubcellOptions");
  TestHelpers::db::test_simple_tag<subcell::Tags::Mesh<Dim>>("Subcell(Mesh)");
  TestHelpers::db::test_compute_tag<subcell::Tags::MeshCompute<Dim>>(
      "Subcell(Mesh)");
  TestHelpers::db::test_simple_tag<
      subcell::Tags::GhostDataForReconstruction<Dim>>(
      "GhostDataForReconstruction");
  TestHelpers::db::test_simple_tag<subcell::Tags::NeighborTciDecisions<Dim>>(
      "NeighborTciDecisions");
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
  TestHelpers::db::test_simple_tag<
      ::Events::Tags::ObserverCoordinates<Dim, Frame::Inertial>>(
      "InertialCoordinates");
  TestHelpers::db::test_simple_tag<
      ::Events::Tags::ObserverCoordinates<Dim, Frame::Grid>>("GridCoordinates");
  TestHelpers::db::test_simple_tag<
      subcell::Tags::CellCenteredFlux<tmpl::list<Var1, Var2>, Dim>>(
      "CellCenteredFlux");
  TestHelpers::db::test_simple_tag<subcell::Tags::CellCenteredFlux<
      tmpl::list<Var1, Var2>, Dim, Frame::Grid>>("CellCenteredFlux");
  TestHelpers::db::test_simple_tag<subcell::Tags::ReconstructionOrder<Dim>>(
      "ReconstructionOrder");
  TestHelpers::db::test_simple_tag<subcell::Tags::MethodOrder<Dim>>(
      "MethodOrder");

  TestHelpers::db::test_compute_tag<
      subcell::Tags::LogicalCoordinatesCompute<Dim>>(
      "ElementLogicalCoordinates");
  TestHelpers::db::test_compute_tag<subcell::Tags::InertialCoordinatesCompute<
      ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                    Frame::Inertial>>>(
      "InertialCoordinates");

  TestHelpers::db::test_compute_tag<
      subcell::Tags::ObserverInverseJacobianCompute<Dim, Frame::ElementLogical,
                                                    Frame::Grid>>(
      "InverseJacobian(ElementLogical,Grid)");
  TestHelpers::db::test_compute_tag<
      subcell::Tags::ObserverInverseJacobianCompute<Dim, Frame::ElementLogical,
                                                    Frame::Inertial>>(
      "InverseJacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_compute_tag<
      subcell::Tags::ObserverInverseJacobianCompute<Dim, Frame::Grid,
                                                    Frame::Inertial>>(
      "InverseJacobian(Grid,Inertial)");

  TestHelpers::db::test_compute_tag<
      subcell::Tags::ObserverJacobianAndDetInvJacobian<
          Dim, Frame::ElementLogical, Frame::Grid>>(
      "Variables(DetInvJacobian(ElementLogical,Grid),Jacobian(ElementLogical,"
      "Grid))");
  TestHelpers::db::test_compute_tag<
      subcell::Tags::ObserverJacobianAndDetInvJacobian<
          Dim, Frame::ElementLogical, Frame::Inertial>>(
      "Variables(DetInvJacobian(ElementLogical,Inertial),Jacobian("
      "ElementLogical,Inertial))");
  TestHelpers::db::test_compute_tag<
      subcell::Tags::ObserverJacobianAndDetInvJacobian<Dim, Frame::Grid,
                                                       Frame::Inertial>>(
      "Variables(DetInvJacobian(Grid,Inertial),Jacobian(Grid,Inertial))");
  TestHelpers::db::test_compute_tag<subcell::Tags::TciStatusCompute<Dim>>(
      "TciStatus");
  TestHelpers::db::test_compute_tag<subcell::Tags::MethodOrderCompute<Dim>>(
      "MethodOrder");

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["translation"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          0.0, std::array<DataVector, 3>{{{Dim, 0.0}, {Dim, -4.3}, {Dim, 0.0}}},
          100.0);
  const auto grid_to_inertial_map =
      moving_mesh
          ? domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
                domain::CoordinateMaps::TimeDependent::Translation<Dim>{
                    "translation"})
          : domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
                domain::CoordinateMaps::Identity<Dim>{});

  const int tci_decision = 22;  // some non-zero value
  const double time = 1.3;
  const Mesh<Dim> dg_mesh(4, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto);
  using ReconsOrder = typename subcell::Tags::ReconstructionOrder<Dim>::type;
  auto active_coords_box = db::create<
      db::AddSimpleTags<domain::Tags::ElementMap<Dim, Frame::Grid>,
                        domain::CoordinateMaps::Tags::CoordinateMap<
                            Dim, Frame::Grid, Frame::Inertial>,
                        domain::Tags::FunctionsOfTimeInitialize, ::Tags::Time,
                        ::domain::Tags::Mesh<Dim>, subcell::Tags::ActiveGrid,
                        subcell::Tags::TciDecision,
                        subcell::Tags::ReconstructionOrder<Dim>,
                        subcell::Tags::SubcellOptions<Dim>>,
      db::AddComputeTags<
          domain::Tags::LogicalCoordinates<Dim>,
          domain::Tags::MappedCoordinates<
              domain::Tags::ElementMap<Dim, Frame::Grid>,
              domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
          domain::Tags::CoordinatesMeshVelocityAndJacobiansCompute<
              domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                          Frame::Inertial>>,
          domain::Tags::InertialFromGridCoordinatesCompute<Dim>,

          subcell::Tags::MeshCompute<Dim>,
          subcell::Tags::LogicalCoordinatesCompute<Dim>,
          subcell::Tags::ObserverMeshCompute<Dim>,

          domain::Tags::MappedCoordinates<
              domain::Tags::ElementMap<Dim, Frame::Grid>,
              evolution::dg::subcell::Tags::Coordinates<Dim,
                                                        Frame::ElementLogical>,
              evolution::dg::subcell::Tags::Coordinates>,
          subcell::Tags::InertialCoordinatesCompute<
              ::domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                            Frame::Inertial>>,

          subcell::Tags::ObserverCoordinatesCompute<Dim, Frame::ElementLogical>,
          subcell::Tags::ObserverCoordinatesCompute<Dim, Frame::Grid>,
          subcell::Tags::ObserverCoordinatesCompute<Dim, Frame::Inertial>,

          subcell::Tags::ObserverInverseJacobianCompute<
              Dim, Frame::ElementLogical, Frame::Grid>,
          subcell::Tags::ObserverInverseJacobianCompute<
              Dim, Frame::ElementLogical, Frame::Inertial>,
          subcell::Tags::ObserverInverseJacobianCompute<Dim, Frame::Grid,
                                                        Frame::Inertial>,
          subcell::Tags::ObserverJacobianAndDetInvJacobian<
              Dim, Frame::ElementLogical, Frame::Grid>,
          subcell::Tags::ObserverJacobianAndDetInvJacobian<
              Dim, Frame::ElementLogical, Frame::Inertial>,
          subcell::Tags::ObserverJacobianAndDetInvJacobian<Dim, Frame::Grid,
                                                           Frame::Inertial>,
          subcell::Tags::TciStatusCompute<Dim>,
          subcell::Tags::MethodOrderCompute<Dim>>>(
      ElementMap<Dim, Frame::Grid>{
          ElementId<Dim>{0},
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<Dim>{})},
      grid_to_inertial_map->get_clone(), clone_unique_ptrs(functions_of_time),
      time, dg_mesh, subcell::ActiveGrid::Dg, tci_decision, ReconsOrder{},
      evolution::dg::subcell::SubcellOptions{
          1.0e-7,
          1.0e-7,
          1.0e-7,
          1.0e-7,
          4.0,
          4.0,
          false,
          evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
          false,
          {},
          ::fd::DerivativeOrder::Two});
  const auto check_box = [&active_coords_box,
                          &tci_decision](const Mesh<Dim>& expected_mesh) {
    (void)tci_decision;  // Incorrect compiler warning.
    CHECK(db::get<::Events::Tags::ObserverMesh<Dim>>(active_coords_box) ==
          expected_mesh);
    const auto expected_logical_coords = logical_coordinates(expected_mesh);
    CHECK(db::get<
              ::Events::Tags::ObserverCoordinates<Dim, Frame::ElementLogical>>(
              active_coords_box) == expected_logical_coords);
    const auto expected_grid_coords =
        db::get<domain::Tags::ElementMap<Dim, Frame::Grid>>(active_coords_box)(
            expected_logical_coords);
    CHECK(db::get<::Events::Tags::ObserverCoordinates<Dim, Frame::Grid>>(
              active_coords_box) == expected_grid_coords);
    const auto expected_inertial_coords =
        db::get<domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                            Frame::Inertial>>(
            active_coords_box)(
            expected_grid_coords, db::get<::Tags::Time>(active_coords_box),
            db::get<domain::Tags::FunctionsOfTime>(active_coords_box));
    CHECK(db::get<::Events::Tags::ObserverCoordinates<Dim, Frame::Inertial>>(
              active_coords_box) == expected_inertial_coords);

    const auto expected_inv_jac_logical_to_grid =
        db::get<domain::Tags::ElementMap<Dim, Frame::Grid>>(active_coords_box)
            .inv_jacobian(expected_logical_coords);
    CHECK(db::get<::Events::Tags::ObserverInverseJacobian<
              Dim, Frame::ElementLogical, Frame::Grid>>(active_coords_box) ==
          expected_inv_jac_logical_to_grid);

    const auto expected_inv_jac_grid_to_inertial =
        db::get<domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                            Frame::Inertial>>(
            active_coords_box)
            .inv_jacobian(
                expected_grid_coords, db::get<::Tags::Time>(active_coords_box),
                db::get<domain::Tags::FunctionsOfTime>(active_coords_box));
    CHECK(db::get<::Events::Tags::ObserverInverseJacobian<Dim, Frame::Grid,
                                                          Frame::Inertial>>(
              active_coords_box) == expected_inv_jac_grid_to_inertial);

    InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
        expected_inv_jac_logical_to_inertial{};
    for (size_t logical_i = 0; logical_i < Dim; ++logical_i) {
      for (size_t inertial_i = 0; inertial_i < Dim; ++inertial_i) {
        expected_inv_jac_logical_to_inertial.get(logical_i, inertial_i) =
            expected_inv_jac_logical_to_grid.get(logical_i, 0) *
            expected_inv_jac_grid_to_inertial.get(0, inertial_i);
        for (size_t grid_i = 1; grid_i < Dim; ++grid_i) {
          expected_inv_jac_logical_to_inertial.get(logical_i, inertial_i) +=
              expected_inv_jac_logical_to_grid.get(logical_i, grid_i) *
              expected_inv_jac_grid_to_inertial.get(grid_i, inertial_i);
        }
      }
    }

    CHECK(db::get<::Events::Tags::ObserverInverseJacobian<
              Dim, Frame::ElementLogical, Frame::Inertial>>(
              active_coords_box) == expected_inv_jac_logical_to_inertial);

    const auto [expected_det_inv_jac_logical_to_grid,
                expected_jac_logical_to_grid] =
        determinant_and_inverse(expected_inv_jac_logical_to_grid);
    const auto [expected_det_inv_jac_grid_to_inertial,
                expected_jac_grid_to_inertial] =
        determinant_and_inverse(expected_inv_jac_grid_to_inertial);
    const auto [expected_det_inv_jac_logical_to_inertial,
                expected_jac_logical_to_inertial] =
        determinant_and_inverse(expected_inv_jac_logical_to_inertial);

    CHECK(db::get<::Events::Tags::ObserverJacobian<Dim, Frame::ElementLogical,
                                                   Frame::Grid>>(
              active_coords_box) == expected_jac_logical_to_grid);
    CHECK(db::get<::Events::Tags::ObserverDetInvJacobian<Frame::ElementLogical,
                                                         Frame::Grid>>(
              active_coords_box) == expected_det_inv_jac_logical_to_grid);

    CHECK(db::get<::Events::Tags::ObserverJacobian<Dim, Frame::Grid,
                                                   Frame::Inertial>>(
              active_coords_box) == expected_jac_grid_to_inertial);
    CHECK(db::get<::Events::Tags::ObserverDetInvJacobian<Frame::Grid,
                                                         Frame::Inertial>>(
              active_coords_box) == expected_det_inv_jac_grid_to_inertial);

    CHECK(db::get<::Events::Tags::ObserverJacobian<Dim, Frame::ElementLogical,
                                                   Frame::Inertial>>(
              active_coords_box) == expected_jac_logical_to_inertial);
    CHECK(db::get<::Events::Tags::ObserverDetInvJacobian<Frame::ElementLogical,
                                                         Frame::Inertial>>(
              active_coords_box) == expected_det_inv_jac_logical_to_inertial);
    CHECK(db::get<subcell::Tags::TciStatus>(active_coords_box) ==
          Scalar<DataVector>(expected_mesh.number_of_grid_points(),
                             static_cast<double>(tci_decision)));
    if (db::get<subcell::Tags::ActiveGrid>(active_coords_box) ==
        subcell::ActiveGrid::Dg) {
      REQUIRE(db::get<subcell::Tags::MethodOrder<Dim>>(active_coords_box)
                  .has_value());
      for (size_t i = 0; i < Dim; ++i) {
        CHECK(db::get<subcell::Tags::MethodOrder<Dim>>(active_coords_box)
                  .value()[i] ==
              DataVector{expected_mesh.number_of_grid_points(),
                         static_cast<double>(expected_mesh.extents(i))});
      }
    } else {
      if (db::get<subcell::Tags::ReconstructionOrder<Dim>>(active_coords_box)
              .has_value()) {
        CHECK(db::get<subcell::Tags::MethodOrder<Dim>>(active_coords_box) ==
              db::get<subcell::Tags::ReconstructionOrder<Dim>>(
                  active_coords_box));
      } else {
        for (size_t i = 0; i < Dim; ++i) {
          CHECK(db::get<subcell::Tags::MethodOrder<Dim>>(active_coords_box)
                    .value()[i] ==
                DataVector{expected_mesh.number_of_grid_points(),
                           static_cast<double>(static_cast<int>(
                               db::get<subcell::Tags::SubcellOptions<Dim>>(
                                   active_coords_box)
                                   .finite_difference_derivative_order()))});
        }
      }
    }
  };

  check_box(db::get<domain::Tags::Mesh<Dim>>(active_coords_box));
  db::mutate<subcell::Tags::ActiveGrid>(
      [](const auto active_grid_ptr) {
        *active_grid_ptr = subcell::ActiveGrid::Subcell;
      },
      make_not_null(&active_coords_box));
  check_box(db::get<subcell::Tags::Mesh<Dim>>(active_coords_box));
  db::mutate<subcell::Tags::ReconstructionOrder<Dim>>(
      [](const auto recons_order_ptr, const size_t num_pts) {
        (*recons_order_ptr) = ReconsOrder{num_pts};
        for (size_t i = 0; i < Dim; ++i) {
          recons_order_ptr->value()[i] = static_cast<double>(i) + 3.0;
        }
      },
      make_not_null(&active_coords_box),
      db::get<subcell::Tags::Mesh<Dim>>(active_coords_box)
          .number_of_grid_points());
  check_box(db::get<subcell::Tags::Mesh<Dim>>(active_coords_box));

  TestHelpers::db::test_compute_tag<subcell::Tags::ObserverMeshCompute<Dim>>(
      "ObserverMesh");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Tags", "[Evolution][Unit]") {
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
  TestHelpers::db::test_simple_tag<subcell::Tags::TciGridHistory>(
      "TciGridHistory");
  TestHelpers::db::test_simple_tag<subcell::Tags::TciStatus>("TciStatus");
  TestHelpers::db::test_simple_tag<subcell::Tags::TciDecision>("TciDecision");

  for (const bool moving_mesh : {false, true}) {
    test<1>(moving_mesh);
    test<2>(moving_mesh);
    test<3>(moving_mesh);
  }
}
