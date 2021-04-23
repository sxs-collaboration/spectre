// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
template <size_t Dim>
void test() {
  const size_t num_pts_1d = 5;
  const Index<Dim> subcell_extents{num_pts_1d};
  for (size_t d = 0; d < Dim; ++d) {
    CAPTURE(d);
    auto extents = make_array<Dim>(num_pts_1d);
    ++gsl::at(extents, d);
    const auto basis = make_array<Dim>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<Dim>(Spectral::Quadrature::CellCentered);
    gsl::at(quadrature, d) = Spectral::Quadrature::FaceCentered;
    const Mesh<Dim> subcell_face_mesh{extents, basis, quadrature};

    DataVector dt_var{subcell_extents.product(), 1.2};
    const DataVector inv_jacobian{subcell_extents.product(), 5.0};
    const auto logical_coords = logical_coordinates(subcell_face_mesh);
    const double one_over_delta =
        1.0 / (get<0>(logical_coords)[1] - get<0>(logical_coords)[0]);
    const DataVector boundary_correction = 3.0 * logical_coords.get(d);
    evolution::dg::subcell::add_cartesian_flux_divergence(
        make_not_null(&dt_var), one_over_delta, inv_jacobian,
        boundary_correction, subcell_extents, d);
    const DataVector expected_dt_var{subcell_extents.product(),
                                     inv_jacobian[0] * 3.0 + 1.2};
    CHECK_ITERABLE_APPROX(dt_var, expected_dt_var);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FD.CartesianFluxDivergence",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
