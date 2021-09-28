// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"

// These tests generate a function `u_nodal_expected` from a linear
// superposition of the basis functions, which are then transformed to spectral
// space (`u_modal`). The coefficients are compared to their expected values,
// which is 1 if the coefficient is specified in `coeffs_to_include` and 0
// otherwise. Finally, the modal coefficients are transformed back to the nodal
// coefficients and compared to `u_nodal_expected`.
namespace {
template <typename ModalVectorType, typename NodalVectorType,
          Spectral::Basis Basis, Spectral::Quadrature Quadrature, size_t Dim>
void check_transforms(
    const Mesh<Dim>& mesh,
    const std::vector<std::array<size_t, Dim>>& coeffs_to_include) {
  CAPTURE(Basis);
  CAPTURE(Quadrature);
  CAPTURE(mesh);
  CAPTURE(coeffs_to_include);

  MAKE_GENERATOR(generator);
  UniformCustomDistribution<
      tt::get_fundamental_type_t<typename ModalVectorType::ElementType>>
      dist{0.5, 2.0};
  const auto basis_factor =
      make_with_random_values<typename ModalVectorType::ElementType>(
          make_not_null(&generator), make_not_null(&dist));

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
  NodalVectorType u_nodal_expected(mesh.number_of_grid_points(), 0.0);
  for (const auto& coeff : coeffs_to_include) {
    // additional * 1.0 in appropriate type necessary for being generic to
    // complex values
    NodalVectorType basis_function =
        basis_factor * Spectral::compute_basis_function_value<Basis>(
                           coeff[0], get<0>(logical_coords));
    for (size_t dim = 1; dim < Dim; ++dim) {
      basis_function *= typename ModalVectorType::ElementType{1.0} *
                        Spectral::compute_basis_function_value<Basis>(
                            gsl::at(coeff, dim), logical_coords.get(dim));
    }
    u_nodal_expected += basis_function;
  }

  // Transform to modal coefficients and check their values
  const ModalVectorType u_modal = to_modal_coefficients(u_nodal_expected, mesh);
  for (IndexIterator<Dim> index_it(mesh.extents()); index_it; ++index_it) {
    CAPTURE(*index_it);
    if (alg::found(coeffs_to_include, index_it->indices())) {
      CHECK_COMPLEX_APPROX(u_modal[index_it.collapsed_index()], basis_factor);
    } else {
      CHECK_COMPLEX_APPROX(u_modal[index_it.collapsed_index()],
                           typename ModalVectorType::ElementType{0.0});
    }
  }

  // Back to nodal coefficients, which should match what we set up initially
  const NodalVectorType u_nodal = to_nodal_coefficients(u_modal, mesh);
  CHECK_ITERABLE_APPROX(u_nodal_expected, u_nodal);
}

template <typename ModalVectorType, typename NodalVectorType,
          Spectral::Basis Basis, Spectral::Quadrature Quadrature,
          typename Generator>
void test_1d(const gsl::not_null<Generator*> generator) {
  UniformCustomDistribution<size_t> dist{
      1, Spectral::maximum_number_of_points<Basis> - 1};
  const size_t order = dist(*generator);
  // Start at 1st order so we are independent of the minimum number of
  // coefficients.
  CAPTURE(order);
  const Mesh<1> mesh(order + 1, Basis, Quadrature);
  check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
      mesh, {{{order}}});
  check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
      mesh, {{{order}}, {{order / 2}}});
  if (order > 4) {
    check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
        mesh, {{{order}}, {{order / 3}}});
  }
}

template <typename ModalVectorType, typename NodalVectorType,
          Spectral::Basis Basis, Spectral::Quadrature Quadrature,
          typename Generator>
void test_2d(const gsl::not_null<Generator*> generator) {
  // Start at one higher order so we can drop one order in the y-direction.
  UniformCustomDistribution<size_t> dist{
      Spectral::minimum_number_of_points<Basis, Quadrature> + 1,
      Spectral::maximum_number_of_points<Basis> - 1};
  const size_t order = dist(*generator);
  CAPTURE(order);
  const Mesh<2> mesh({{order + 1, order}}, Basis, Quadrature);
  check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
      mesh, {{{order, order - 1}}});
  check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
      mesh, {{{order, order - 1}}, {{order / 2, order / 2}}});
  check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
      Mesh<2>{{{order + 1, order + 1}}, Basis, Quadrature},
      {{{order - 1, order}}, {{order / 2, order / 2}}});
  check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
      Mesh<2>{{{order + 1, order + 1}}, Basis, Quadrature},
      {{{order, order}}, {{order / 2, order / 2}}});
  check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
      mesh, {{{order, order - 1}}, {{order / 3, order / 3}}});
}

template <typename ModalVectorType, typename NodalVectorType,
          Spectral::Basis Basis, Spectral::Quadrature Quadrature,
          typename Generator>
void test_3d(const gsl::not_null<Generator*> generator) {
  // Start at two orders higher so we can drop one order in the y-direction and
  // two in z-direction.
  UniformCustomDistribution<size_t> dist{
      Spectral::minimum_number_of_points<Basis, Quadrature> + 2,
      Spectral::maximum_number_of_points<Basis> - 2};
  const size_t order = dist(*generator);
  CAPTURE(order);
  const Mesh<3> mesh({{order + 1, order, order - 1}}, Basis, Quadrature);
  check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
      mesh, {{{order, order - 1, order - 2}}});
  check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
      mesh,
      {{{order, order - 1, order - 2}}, {{order / 2, order / 2, order / 2}}});
  check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
      Mesh<3>{{{order + 1, order + 1, order + 1}}, Basis, Quadrature},
      {{{order, order, order}}});
  check_transforms<ModalVectorType, NodalVectorType, Basis, Quadrature>(
      mesh,
      {{{order, order - 1, order - 2}}, {{order / 3, order / 3, order / 3}}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.CoefficientTransforms",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  MAKE_GENERATOR(generator);
  test_1d<ModalVector, DataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_1d<ModalVector, DataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));
  test_2d<ModalVector, DataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_2d<ModalVector, DataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));
  test_3d<ModalVector, DataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_3d<ModalVector, DataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));

  test_1d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_1d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));
  test_2d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_2d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));
  test_3d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_3d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Legendre,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));

  test_1d<ModalVector, DataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_1d<ModalVector, DataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));
  test_2d<ModalVector, DataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_2d<ModalVector, DataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));
  test_3d<ModalVector, DataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_3d<ModalVector, DataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));

  test_1d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_1d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));
  test_2d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_2d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));
  test_3d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::GaussLobatto>(make_not_null(&generator));
  test_3d<ComplexModalVector, ComplexDataVector, Spectral::Basis::Chebyshev,
          Spectral::Quadrature::Gauss>(make_not_null(&generator));
}
