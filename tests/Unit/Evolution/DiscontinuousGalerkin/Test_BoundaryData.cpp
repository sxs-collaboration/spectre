// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GetOutput.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim>
void test() {
  CAPTURE(Dim);
  const Mesh<Dim> volume_mesh{5, Spectral::Basis::Legendre,
                              Spectral::Quadrature::Gauss};
  const Mesh<Dim> ghost_data_mesh{9, Spectral::Basis::FiniteDifference,
                              Spectral::Quadrature::CellCentered};
  const Mesh<Dim - 1> interface_mesh = volume_mesh.slice_away(0);
  const Time time{{0.0, 1.0}, {0, 1}};
  const BoundaryData<Dim> data0{volume_mesh,
                                ghost_data_mesh,
                                interface_mesh,
                                DataVector{2, 2.3},
                                DataVector{1, 4.4},
                                TimeStepId{true, 1, time},
                                7,
                                3};
  CHECK(data0 == BoundaryData<Dim>{volume_mesh, ghost_data_mesh, interface_mesh,
                                   DataVector{2, 2.3}, DataVector{1, 4.4},
                                   TimeStepId{true, 1, time}, 7, 3});
  CHECK(data0 != BoundaryData<Dim>{Mesh<Dim>{6, Spectral::Basis::Legendre,
                                             Spectral::Quadrature::Gauss},
                                   ghost_data_mesh, interface_mesh,
                                   DataVector{2, 2.3}, DataVector{1, 4.4},
                                   TimeStepId{true, 1, time}, 7, 3});
  CHECK(data0 !=
        BoundaryData<Dim>{volume_mesh,
                          Mesh<Dim>{11, Spectral::Basis::FiniteDifference,
                                    Spectral::Quadrature::CellCentered},
                          interface_mesh, DataVector{2, 2.3},
                          DataVector{1, 4.4}, TimeStepId{true, 1, time}, 7, 3});
  if constexpr (Dim > 1) {
    CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh,
                                     Mesh<Dim - 1>{2, Spectral::Basis::Legendre,
                                                   Spectral::Quadrature::Gauss},
                                     DataVector{2, 2.3}, DataVector{1, 4.4},
                                     TimeStepId{true, 1, time}, 7, 3});
  }
  CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh, interface_mesh,
                                   DataVector{9, 2.3}, DataVector{1, 4.4},
                                   TimeStepId{true, 1, time}, 7, 3});
  CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh, interface_mesh,
                                   DataVector{2, 2.3}, DataVector{6, 4.4},
                                   TimeStepId{true, 1, time}, 7, 3});
  CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh, interface_mesh,
                                   DataVector{2, 2.3}, DataVector{1, 4.4},
                                   TimeStepId{true, 2, time}, 7, 3});
  CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh, interface_mesh,
                                   DataVector{2, 2.3}, DataVector{1, 4.4},
                                   TimeStepId{true, 1, time}, 9, 3});
  CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh, interface_mesh,
                                   DataVector{2, 2.3}, DataVector{1, 4.4},
                                   TimeStepId{true, 2, time}, 7, 5});
  CHECK(get_output(data0) ==
        std::string("Volume mesh: " + get_output(volume_mesh) +
                    "\nGhost mesh: " + get_output(ghost_data_mesh) +
                    "\nInterface mesh: " + get_output(interface_mesh) +
                    "\nGhost cell data: " + get_output(DataVector{2, 2.3}) +
                    "\nBoundary correction: " + get_output(DataVector{1, 4.4}) +
                    "\nValidy range: " + get_output(TimeStepId{true, 1, time}) +
                    "\nTCI status: 7\nIntegration order: 3"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.BoundaryData", "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace evolution::dg
