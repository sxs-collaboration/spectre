// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Tags {
struct Scalar : db::SimpleTag {
  using type = ::Scalar<DataVector>;
};

template <size_t Dim>
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};
}  // namespace Tags

template <size_t Dim>
DataVector soln(const tnsr::I<DataVector, Dim, Frame::Logical>& coords,
                const bool add_discontinuity) noexcept {
  DataVector result =
      Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
          1, get<0>(coords));
  for (size_t d = 1; d < Dim; ++d) {
    result += Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
        1, coords.get(d));
  }

  if (add_discontinuity) {
    const double max_value = max(abs(result));
    for (size_t i = 0; i < result.size() / 2; ++i) {
      result[i] += 10.0 * max_value;
    }
  }
  return result;
}

template <size_t Dim>
void test_two_mesh_rdmp_impl(const size_t num_pts_1d,
                             const size_t tensor_component_to_modify,
                             const double rdmp_delta0,
                             const double rdmp_epsilon,
                             const bool expected_tci_triggered) noexcept {
  CAPTURE(Dim);
  CAPTURE(num_pts_1d);
  CAPTURE(tensor_component_to_modify);
  CAPTURE(rdmp_delta0);
  CAPTURE(rdmp_epsilon);
  CAPTURE(expected_tci_triggered);
  const Mesh<Dim> dg_mesh{num_pts_1d, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh{2 * num_pts_1d - 1,
                               Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::FaceCentered};
  const auto logical_coords = logical_coordinates(dg_mesh);

  Variables<tmpl::list<Tags::Scalar, Tags::Vector<Dim>>> dg_vars(
      dg_mesh.number_of_grid_points());
  get(get<Tags::Scalar>(dg_vars)) =
      soln(logical_coords, tensor_component_to_modify == 0);
  for (size_t d = 0; d < Dim; ++d) {
    get<Tags::Vector<Dim>>(dg_vars).get(d) =
        (d + 0.3) * soln(logical_coords, tensor_component_to_modify == d + 1);
  }

  const Variables<
      tmpl::list<evolution::dg::subcell::Tags::Inactive<Tags::Scalar>,
                 evolution::dg::subcell::Tags::Inactive<Tags::Vector<Dim>>>>
      subcell_vars{evolution::dg::subcell::fd::project(dg_vars, dg_mesh,
                                                       subcell_mesh.extents())};

  CHECK(evolution::dg::subcell::two_mesh_rdmp_tci(dg_vars, subcell_vars,
                                                  rdmp_delta0, rdmp_epsilon) ==
        expected_tci_triggered);
}

template <size_t Dim>
void test_two_mesh_rdmp() noexcept {
  // We lower the maximum number of 1d points in 3d in order to reduce total
  // test runtime.
  const size_t maximum_number_of_points_1d =
      Dim == 3 ? 7
               : Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
  for (size_t num_pts_1d = 4; num_pts_1d < maximum_number_of_points_1d;
       ++num_pts_1d) {
    test_two_mesh_rdmp_impl<Dim>(num_pts_1d, 0, 1.0e-4, 1.0e-3, true);
    test_two_mesh_rdmp_impl<Dim>(num_pts_1d, 0, 1.0e-4, 1.0e3, false);
    test_two_mesh_rdmp_impl<Dim>(num_pts_1d, std::numeric_limits<size_t>::max(),
                                 1.0e-4, 1.0e-3, false);

    // Modify tensor components
    for (size_t i = 1; i < Dim + 1; ++i) {
      test_two_mesh_rdmp_impl<Dim>(num_pts_1d, i, 1.0e-4, 1.0e-3, true);
      test_two_mesh_rdmp_impl<Dim>(num_pts_1d, i, 1.0e-4, 1.0e3, false);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Tci.TwoMeshRdmp",
                  "[Evolution][Unit]") {
  test_two_mesh_rdmp<1>();
  test_two_mesh_rdmp<2>();
  test_two_mesh_rdmp<3>();
}
}  // namespace
