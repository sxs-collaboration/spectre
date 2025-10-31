// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/InterpolatedBoundaryData.hpp"
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
  const Mesh<Dim - 1> mortar_mesh{6, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::Gauss};
  const Time time{{0.0, 1.0}, {0, 1}};
  const BoundaryData<Dim> data0{volume_mesh,
                                ghost_data_mesh,
                                mortar_mesh,
                                DataVector{2, 2.3},
                                DataVector{1, 4.4},
                                TimeStepId{true, 1, time},
                                7,
                                3,
                                std::nullopt};
  CHECK(data0 == BoundaryData<Dim>{volume_mesh, ghost_data_mesh, mortar_mesh,
                                   DataVector{2, 2.3}, DataVector{1, 4.4},
                                   TimeStepId{true, 1, time}, 7, 3,
                                   std::nullopt});
  CHECK(
      data0 !=
      BoundaryData<Dim>{
          Mesh<Dim>{6, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
          ghost_data_mesh, mortar_mesh, DataVector{2, 2.3}, DataVector{1, 4.4},
          TimeStepId{true, 1, time}, 7, 3, std::nullopt});
  CHECK(data0 !=
        BoundaryData<Dim>{volume_mesh,
                          Mesh<Dim>{11, Spectral::Basis::FiniteDifference,
                                    Spectral::Quadrature::CellCentered},
                          mortar_mesh, DataVector{2, 2.3}, DataVector{1, 4.4},
                          TimeStepId{true, 1, time}, 7, 3, std::nullopt});
  if constexpr (Dim > 1) {
    CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh,
                                     Mesh<Dim - 1>{2, Spectral::Basis::Legendre,
                                                   Spectral::Quadrature::Gauss},
                                     DataVector{2, 2.3}, DataVector{1, 4.4},
                                     TimeStepId{true, 1, time}, 7, 3,
                                     std::nullopt});
  }
  CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh, mortar_mesh,
                                   DataVector{9, 2.3}, DataVector{1, 4.4},
                                   TimeStepId{true, 1, time}, 7, 3,
                                   std::nullopt});
  CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh, mortar_mesh,
                                   DataVector{2, 2.3}, DataVector{6, 4.4},
                                   TimeStepId{true, 1, time}, 7, 3,
                                   std::nullopt});
  CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh, mortar_mesh,
                                   DataVector{2, 2.3}, DataVector{1, 4.4},
                                   TimeStepId{true, 2, time}, 7, 3,
                                   std::nullopt});
  CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh, mortar_mesh,
                                   DataVector{2, 2.3}, DataVector{1, 4.4},
                                   TimeStepId{true, 1, time}, 9, 3,
                                   std::nullopt});
  CHECK(data0 != BoundaryData<Dim>{volume_mesh, ghost_data_mesh, mortar_mesh,
                                   DataVector{2, 2.3}, DataVector{1, 4.4},
                                   TimeStepId{true, 2, time}, 7, 5,
                                   std::nullopt});
  CHECK(data0 !=
        BoundaryData<Dim>{volume_mesh, ghost_data_mesh, mortar_mesh,
                          DataVector{2, 2.3}, DataVector{1, 4.4},
                          TimeStepId{true, 1, time}, 7, 3,
                          InterpolatedBoundaryData<Dim>{
                              {.data = DataVector{3, 1.0},
                               .target_mesh = mortar_mesh,
                               .offsets = std::vector{0_st, 2_st, 3_st}}}});
  CHECK(get_output(data0) ==
        std::string(
            "Volume mesh: " + get_output(volume_mesh) +
            "\nGhost mesh: " + get_output(ghost_data_mesh) +
            "\nBoundary correction mesh: " + get_output(mortar_mesh) +
            "\nGhost cell data: " + get_output(DataVector{2, 2.3}) +
            "\nBoundary correction data: " + get_output(DataVector{1, 4.4}) +
            "\nValidy range: " + get_output(TimeStepId{true, 1, time}) +
            "\nTCI status: 7\nIntegration order: 3\nInterpolated boundary "
            "data: --"));

  // Test merge_boundary_data
  const BoundaryData<Dim> dg_data{
      volume_mesh,        ghost_data_mesh,           mortar_mesh, std::nullopt,
      DataVector{1, 4.4}, TimeStepId{true, 1, time}, 7,           3,
      std::nullopt};
  {
    BoundaryData<Dim> ghost_data{volume_mesh,
                                 ghost_data_mesh,
                                 std::nullopt,
                                 DataVector{2, 2.3},
                                 std::nullopt,
                                 TimeStepId{true, 1, time},
                                 0,
                                 0,
                                 std::nullopt};
    merge_boundary_data(make_not_null(&ghost_data), dg_data);
    CHECK(ghost_data == data0);
  }
#ifdef SPECTRE_DEBUG
  {
    BoundaryData<Dim> ghost_data{ghost_data_mesh,
                                 ghost_data_mesh,
                                 std::nullopt,
                                 DataVector{2, 2.3},
                                 std::nullopt,
                                 TimeStepId{true, 1, time},
                                 0,
                                 0,
                                 std::nullopt};
    CHECK_THROWS_WITH(
        merge_boundary_data(make_not_null(&ghost_data), dg_data),
        Catch::Matchers::ContainsSubstring(
            "The mesh being received for the fluxes is different"));
  }
  {
    BoundaryData<Dim> ghost_data{volume_mesh,
                                 volume_mesh,
                                 std::nullopt,
                                 DataVector{2, 2.3},
                                 std::nullopt,
                                 TimeStepId{true, 1, time},
                                 0,
                                 0,
                                 std::nullopt};
    CHECK_THROWS_WITH(
        merge_boundary_data(make_not_null(&ghost_data), dg_data),
        Catch::Matchers::ContainsSubstring(
            "The mesh being received for the ghost cell data is different"));
  }
  {
    BoundaryData<Dim> ghost_data{volume_mesh,
                                 ghost_data_mesh,
                                 mortar_mesh,
                                 DataVector{2, 2.3},
                                 std::nullopt,
                                 TimeStepId{true, 1, time},
                                 0,
                                 0,
                                 std::nullopt};
    CHECK_THROWS_WITH(merge_boundary_data(make_not_null(&ghost_data), dg_data),
                      Catch::Matchers::ContainsSubstring(
                          "The fluxes have already been received"));
  }
  {
    BoundaryData<Dim> ghost_data{volume_mesh,
                                 ghost_data_mesh,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                                 TimeStepId{true, 1, time},
                                 0,
                                 0,
                                 std::nullopt};
    CHECK_THROWS_WITH(merge_boundary_data(make_not_null(&ghost_data), dg_data),
                      Catch::Matchers::ContainsSubstring(
                          "Have not yet received ghost cells"));
  }
  {
    BoundaryData<Dim> ghost_data{volume_mesh,
                                 ghost_data_mesh,
                                 std::nullopt,
                                 DataVector{2, 2.3},
                                 DataVector{1, 4.4},
                                 TimeStepId{true, 1, time},
                                 0,
                                 0,
                                 std::nullopt};
    CHECK_THROWS_WITH(merge_boundary_data(make_not_null(&ghost_data), dg_data),
                      Catch::Matchers::ContainsSubstring(
                          "The fluxes have already been received"));
  }
#endif  // SPECTRE_DEBUG
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.BoundaryData", "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace evolution::dg
