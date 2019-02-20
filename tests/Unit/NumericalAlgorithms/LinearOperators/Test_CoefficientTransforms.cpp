// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"

// These tests generate a function `u_nodal_expected` from a linear
// superposition of the basis functions, which are then transformed to spectral
// space (`u_modal`). The coefficients are compared to their expected values,
// which is 1 if the coefficient is specified in `coeffs_to_include` and 0
// otherwise. Finally, the modal coefficients are transformed back to the nodal
// coefficients and compared to `u_nodal_expected`.
namespace {
template <Spectral::Basis Basis, Spectral::Quadrature Quadrature, size_t Dim>
void check_transforms(
    const Mesh<Dim>& mesh,
    const std::vector<std::array<size_t, Dim>>& coeffs_to_include) noexcept {
  CAPTURE(Basis);
  CAPTURE(Quadrature);
  CAPTURE(mesh);
  CAPTURE(coeffs_to_include);

  // Construct a functions:
  //
  // 1D: u(\xi) = sum_i c_i phi_i(\xi)
  // 2D: u(\xi,\eta) = sum_{i,j} c_{i,j} phi_i(\xi) phi_j(\eta)
  // 3D: u(\xi,\eta,\zeta) = sum_{i,j,k} c_{i,j,k} phi_i(\xi)
  //                                     phi_j(\eta) phi_k(\zeta)
  //
  // where the coefficients c_{i,j,k} are 1 if listed in `coeffs_to_include` and
  // 0 otherwise.
  const auto logical_coords = logical_coordinates(mesh);
  DataVector u_nodal_expected(mesh.number_of_grid_points(), 0.0);
  for (const auto& coeff : coeffs_to_include) {
    DataVector basis_function = Spectral::compute_basis_function_value<Basis>(
        coeff[0], get<0>(logical_coords));
    for (size_t dim = 1; dim < Dim; ++dim) {
      basis_function *= Spectral::compute_basis_function_value<Basis>(
          gsl::at(coeff, dim), logical_coords.get(dim));
    }
    u_nodal_expected += basis_function;
  }

  // Transform to modal coefficients and check their values
  const ModalVector u_modal = to_modal_coefficients(u_nodal_expected, mesh);
  for (IndexIterator<Dim> index_it(mesh.extents()); index_it; ++index_it) {
    CAPTURE(*index_it);
    if (alg::found(coeffs_to_include, index_it->indices())) {
      CHECK(u_modal[index_it.collapsed_index()] == approx(1.0));
    } else {
      CHECK(u_modal[index_it.collapsed_index()] == approx(0.0));
    }
  }

  // Back to nodal coefficients, which should match what we set up initially
  const DataVector u_nodal = to_nodal_coefficients(u_modal, mesh);
  CHECK_ITERABLE_APPROX(u_nodal_expected, u_nodal);
}

template <Spectral::Basis Basis, Spectral::Quadrature Quadrature>
void test_1d() noexcept {
  // Start at 1st order so we are independent of the minimum number of
  // coefficients.
  for (size_t order = 1; order < Spectral::maximum_number_of_points<Basis>;
       ++order) {
    CAPTURE(order);
    const Mesh<1> mesh(order + 1, Basis, Quadrature);
    check_transforms<Basis, Quadrature>(mesh, {{{{order}}}});
    check_transforms<Basis, Quadrature>(mesh, {{{{order}}, {{order / 2}}}});
    if (order > 4) {
      check_transforms<Basis, Quadrature>(mesh, {{{{order}}, {{order / 3}}}});
    }
  }
}

template <Spectral::Basis Basis, Spectral::Quadrature Quadrature>
void test_2d() noexcept {
  // Start at one higher order so we can drop one order in the y-direction.
  for (size_t order = Spectral::minimum_number_of_points<Basis, Quadrature> + 1;
       order < Spectral::maximum_number_of_points<Basis>; ++order) {
    CAPTURE(order);
    const Mesh<2> mesh({{order + 1, order}}, Basis, Quadrature);
    check_transforms<Basis, Quadrature>(mesh, {{{{order, order - 1}}}});
    check_transforms<Basis, Quadrature>(
        mesh, {{{{order, order - 1}}, {{order / 2, order / 2}}}});
    check_transforms<Basis, Quadrature>(
        Mesh<2>{{{order + 1, order + 1}}, Basis, Quadrature},
        {{{{order - 1, order}}, {{order / 2, order / 2}}}});
    check_transforms<Basis, Quadrature>(
        Mesh<2>{{{order + 1, order + 1}}, Basis, Quadrature},
        {{{{order, order}}, {{order / 2, order / 2}}}});
    check_transforms<Basis, Quadrature>(
        mesh, {{{{order, order - 1}}, {{order / 3, order / 3}}}});
  }
}

template <Spectral::Basis Basis, Spectral::Quadrature Quadrature>
void test_3d() noexcept {
  // Start at two orders higher so we can drop one order in the y-direction and
  // two in z-direction.
  for (size_t order = Spectral::minimum_number_of_points<Basis, Quadrature> + 2;
       order < Spectral::maximum_number_of_points<Basis>; ++order) {
    CAPTURE(order);
    const Mesh<3> mesh({{order + 1, order, order - 1}}, Basis, Quadrature);
    check_transforms<Basis, Quadrature>(mesh,
                                        {{{{order, order - 1, order - 2}}}});
    check_transforms<Basis, Quadrature>(
        mesh, {{{{order, order - 1, order - 2}},
                {{order / 2, order / 2, order / 2}}}});
    check_transforms<Basis, Quadrature>(
        Mesh<3>{{{order + 1, order + 1, order + 1}}, Basis, Quadrature},
        {{{{order, order, order}}}});
    check_transforms<Basis, Quadrature>(
        mesh, {{{{order, order - 1, order - 2}},
                {{order / 3, order / 3, order / 3}}}});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.CoefficientTransforms",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_1d<Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>();
  test_1d<Spectral::Basis::Legendre, Spectral::Quadrature::Gauss>();
  test_2d<Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>();
  test_2d<Spectral::Basis::Legendre, Spectral::Quadrature::Gauss>();
  test_3d<Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>();
  test_3d<Spectral::Basis::Legendre, Spectral::Quadrature::Gauss>();

  test_1d<Spectral::Basis::Chebyshev, Spectral::Quadrature::GaussLobatto>();
  test_1d<Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss>();
  test_2d<Spectral::Basis::Chebyshev, Spectral::Quadrature::GaussLobatto>();
  test_2d<Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss>();
  test_3d<Spectral::Basis::Chebyshev, Spectral::Quadrature::GaussLobatto>();
  test_3d<Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss>();
}
