// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <deque>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/TciStatus.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace evolution::dg::subcell {
namespace {
template <size_t Dim>
void test() {
  const Mesh<Dim> dg_mesh{3, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh{5, Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
  const Scalar<DataVector> dg_true{dg_mesh.number_of_grid_points(), 1.0};
  const Scalar<DataVector> dg_false{dg_mesh.number_of_grid_points(), 0.0};
  const Scalar<DataVector> subcell_true{subcell_mesh.number_of_grid_points(),
                                        1.0};
  const Scalar<DataVector> subcell_false{subcell_mesh.number_of_grid_points(),
                                         0.0};

  CHECK(dg_false ==
        subcell::tci_status(dg_mesh, subcell_mesh, ActiveGrid::Dg, {}));
  CHECK(dg_false ==
        tci_status(dg_mesh, subcell_mesh, ActiveGrid::Dg, {ActiveGrid::Dg}));
  CHECK(dg_false == tci_status(dg_mesh, subcell_mesh, ActiveGrid::Dg,
                               {ActiveGrid::Dg, ActiveGrid::Subcell}));
  CHECK(dg_true == tci_status(dg_mesh, subcell_mesh, ActiveGrid::Dg,
                              {ActiveGrid::Subcell, ActiveGrid::Dg}));
  CHECK(dg_true == tci_status(dg_mesh, subcell_mesh, ActiveGrid::Dg,
                              {ActiveGrid::Subcell}));

  CHECK(subcell_true ==
        tci_status(dg_mesh, subcell_mesh, ActiveGrid::Subcell, {}));
  CHECK(subcell_false == tci_status(dg_mesh, subcell_mesh, ActiveGrid::Subcell,
                                    {ActiveGrid::Dg}));
  CHECK(subcell_false == tci_status(dg_mesh, subcell_mesh, ActiveGrid::Subcell,
                                    {ActiveGrid::Dg, ActiveGrid::Subcell}));
  CHECK(subcell_true == tci_status(dg_mesh, subcell_mesh, ActiveGrid::Subcell,
                                   {ActiveGrid::Subcell, ActiveGrid::Dg}));
  CHECK(subcell_true == tci_status(dg_mesh, subcell_mesh, ActiveGrid::Subcell,
                                   {ActiveGrid::Subcell}));
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.TciStatus", "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
}  // namespace evolution::dg::subcell
