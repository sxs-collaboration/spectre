// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/LinearSolve.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/LinearOperators/IndefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/Transpose.hpp"
#include "NumericalAlgorithms/LinearSolver/Lapack.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/StaticCache.hpp"
#include "Utilities/VectorAlgebra.hpp"

namespace Cce {
namespace {
// This builds up the spectral representation of the matrix associated with the
// linear operator (1 - y) d_y f + 2 f.
// Broadly, this is accomplished by manipulating both the right and left hand
// sides of the Legendre identity:
// (2n + 1) P_n = d_x (P_{n + 1} - P_{n - 1})
// Writing this as operations on the modal coefficients,
//  \sum_n (A * m)_n P_n = \sum_n (B * m)_n d_x P_n
// So, in particular, we can take advantage of this to obtain a formula for the
// coefficients of the integral given coefficients m_1 of the input:
//   \sum_n (m_1)_n P_n = \sum_n (B * A^{-1} * m_1)_n d_x P_n,
// Therefore the matrix we wish to act with is B * A^{-1}.
// In this function we calculate the matrix M such that
//   \sum_n (m_2)_n P_n = \sum_n (M * m_2) ((1 - x) d_x P_n  + 2 * P_n)
// Using
//   \sum_n  m_n  (1 + x) P_n  = \sum_n (C * m)_n P_n,
// so
//  \sum_n (C * A * m)_n P_n = \sum_n (B * m)_n ((1 - x) d_x P_n)
// Adding \sum_n (2* B * m_n P_n) to both sides,
//   \sum_n ((C * A + 2 B ) * m)_n P_n
//        = \sum_n (B * m)_n ((1 - x) d_x P_n + 2 * P_n)
// Therefore, the matrix M = B * (C * A + 2 B)^{-1} .
Matrix q_integration_matrix(const size_t number_of_points) noexcept {
  Matrix inverse_one_minus_y = Matrix(number_of_points, number_of_points, 0.0);
  for (size_t i = 1; i < number_of_points - 1; ++i) {
    inverse_one_minus_y(i, i - 1) = i / -(2.0 * i - 1.0);
    inverse_one_minus_y(i, i) = 1.0;
    inverse_one_minus_y(i, i + 1) = (i + 1.0) / -(2.0 * i + 3.0);
  }
  inverse_one_minus_y(0, 0) = 1.0;
  inverse_one_minus_y(0, 1) = -1.0 / 3.0;
  inverse_one_minus_y(number_of_points - 1, number_of_points - 2) =
      -(number_of_points - 1.0) / (2.0 * (number_of_points - 1.0) - 1.0);
  inverse_one_minus_y(number_of_points - 1, number_of_points - 1) = 1.0;

  Matrix indefinite_integral(number_of_points, number_of_points, 0.0);
  for (size_t i = 1; i < number_of_points - 1; ++i) {
    indefinite_integral(i, i - 1) = 1.0;
    indefinite_integral(i, i + 1) = -1.0;
  }
  indefinite_integral(0, 1) = -1.0;
  indefinite_integral(number_of_points - 1, number_of_points - 2) = 1.0;

  Matrix dy_identity_lhs(number_of_points, number_of_points, 0.0);
  for (size_t i = 0; i < number_of_points - 1; ++i) {
    dy_identity_lhs(i, i) = 2.0 * i + 1.0;
  }

  Matrix lhs_mat = inverse_one_minus_y * dy_identity_lhs;

  for (size_t i = 1; i < number_of_points - 1; ++i) {
    lhs_mat(i, i - 1) += 2.0;
    lhs_mat(i, i + 1) += -2.0;
  }
  lhs_mat(0, 1) += -2.0;
  lhs_mat(number_of_points - 1, number_of_points - 2) += 2.0;

  return Spectral::modal_to_nodal_matrix<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
             number_of_points) *
         indefinite_integral * inv(lhs_mat) *
         Spectral::nodal_to_modal_matrix<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
             number_of_points);
}
}  // namespace

const Matrix& precomputed_cce_q_integrator(
    const size_t number_of_radial_grid_points) noexcept {
  static const auto lazy_matrix_cache = make_static_cache<CacheRange<
      1, Spectral::maximum_number_of_points<Spectral::Basis::Legendre> + 1>>(
      [](const size_t local_number_of_radial_points) noexcept {
        return q_integration_matrix(local_number_of_radial_points);
      });
  return lazy_matrix_cache(number_of_radial_grid_points);
}

void radial_integrate_cce_pole_equations(
    const gsl::not_null<ComplexDataVector*> integral_result,
    const ComplexDataVector& pole_of_integrand,
    const ComplexDataVector& regular_integrand,
    const ComplexDataVector& boundary, const ComplexDataVector& one_minus_y,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  const ComplexDataVector integrand =
      pole_of_integrand + one_minus_y * regular_integrand;

  apply_matrices(integral_result,
                 std::array<Matrix, 3>{
                     {Matrix{}, Matrix{},
                      precomputed_cce_q_integrator(number_of_radial_points)}},
                 integrand,
                 Spectral::Swsh::swsh_volume_mesh_for_radial_operations(
                     l_max, number_of_radial_points)
                     .extents());

  // apply boundary condition
  const ComplexDataVector boundary_correction =
      0.25 * (boundary -
              ComplexDataVector{
                  integral_result->data(),
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max)});
  const ComplexDataVector one_minus_y_squared = square(
      1.0 -
      std::complex<double>(1.0, 0.0) *
          Spectral::collocation_points<Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto>(
              number_of_radial_points));
  *integral_result += outer_product(boundary_correction, one_minus_y_squared);
}

namespace detail {
void transpose_to_reals_then_imags_radial_stripes(
    const gsl::not_null<DataVector*> result, const ComplexDataVector& input,
    const size_t number_of_radial_points,
    const size_t number_of_angular_points) noexcept {
  for (size_t i = 0; i < input.size() * 2; ++i) {
    (*result)[i] = ((i / number_of_radial_points) % 2) == 0
                       ? real(input[number_of_angular_points *
                                        (i % number_of_radial_points) +
                                    i / (2 * number_of_radial_points)])
                       : imag(input[number_of_angular_points *
                                        (i % number_of_radial_points) +
                                    i / (2 * number_of_radial_points)]);
  }
}
}  // namespace detail

// generic template applies to `Tags::BondiBeta` and `Tags::BondiU`
template <template <typename> class BoundaryPrefix, typename Tag>
void RadialIntegrateBondi<BoundaryPrefix, Tag>::apply(
    const gsl::not_null<Scalar<
        SpinWeighted<ComplexDataVector, db::item_type<Tag>::type::spin>>*>
        integral_result,
    const Scalar<SpinWeighted<ComplexDataVector,
                              db::item_type<Tag>::type::spin>>& integrand,
    const Scalar<SpinWeighted<ComplexDataVector,
                              db::item_type<Tag>::type::spin>>& boundary,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  indefinite_integral(make_not_null(&get(*integral_result).data()),
                      get(integrand).data(),
                      Spectral::Swsh::swsh_volume_mesh_for_radial_operations(
                          l_max, number_of_radial_points),
                      2);
  // add in the boundary data to each angular slice
  for (size_t i = 0; i < number_of_radial_points; i++) {
    ComplexDataVector angular_view{
        get(*integral_result).data().data() +
            Spectral::Swsh::number_of_swsh_collocation_points(l_max) * i,
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    angular_view += get(boundary).data();
  }
}

template <template <typename> class BoundaryPrefix>
void RadialIntegrateBondi<BoundaryPrefix, Tags::BondiQ>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*>
        integral_result,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& pole_of_integrand,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& regular_integrand,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& boundary,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  radial_integrate_cce_pole_equations(
      make_not_null(&get(*integral_result).data()),
      get(pole_of_integrand).data(), get(regular_integrand).data(),
      get(boundary).data(), get(one_minus_y).data(), l_max,
      number_of_radial_points);
}

template <template <typename> class BoundaryPrefix>
void RadialIntegrateBondi<BoundaryPrefix, Tags::BondiW>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        integral_result,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& pole_of_integrand,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& regular_integrand,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  radial_integrate_cce_pole_equations(
      make_not_null(&get(*integral_result).data()),
      get(pole_of_integrand).data(), get(regular_integrand).data(),
      get(boundary).data(), get(one_minus_y).data(), l_max,
      number_of_radial_points);
}

template <template <typename> class BoundaryPrefix>
void RadialIntegrateBondi<BoundaryPrefix, Tags::BondiH>::apply(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*>
        integral_result,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& pole_of_integrand,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& regular_integrand,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& linear_factor,
    const Scalar<SpinWeighted<ComplexDataVector, 4>>&
        linear_factor_of_conjugate,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& one_minus_y,
    const size_t l_max, const size_t number_of_radial_points) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Matrix operator_matrix(2 * number_of_radial_points,
                         2 * number_of_radial_points);

  ComplexDataVector integrand =
      get(pole_of_integrand).data() +
      get(one_minus_y).data() * get(regular_integrand).data();

  DataVector transpose_buffer{2 * get(pole_of_integrand).size()};
  DataVector linear_solve_buffer{2 * get(pole_of_integrand).size()};

  // transpose such that each radial slice is split up into the order:
  // (real radial slice 00) (imag radial slice 00) (real radial slice 01) ...
  detail::transpose_to_reals_then_imags_radial_stripes(
      make_not_null(&linear_solve_buffer), integrand, number_of_radial_points,
      number_of_angular_points);

  raw_transpose(make_not_null(transpose_buffer.data()),
                linear_solve_buffer.data(), number_of_radial_points,
                2 * number_of_angular_points);

  const auto& derivative_matrix =
      Spectral::differentiation_matrix<Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto>(
          number_of_radial_points);
  for (size_t offset = 0; offset < number_of_angular_points; ++offset) {
    // on repeated evaluations, the matrix gets permuted by the dgesv routine.
    // We'll ignore its pivots and just overwrite the whole thing on each
    // pass. There are probably optimizations that can be made which make use
    // of the pivots.

    // first we apply the (1 - y) \partial_y part of the matrix
    // to the upper right (real-real) and lower left (imag-imag) part of the
    // matrix
    for (size_t matrix_block = 0; matrix_block < 2; ++matrix_block) {
      for (size_t i = 0; i < number_of_radial_points; ++i) {
        for (size_t j = 0; j < number_of_radial_points; ++j) {
          operator_matrix(i + matrix_block * number_of_radial_points,
                          j + matrix_block * number_of_radial_points) =
              derivative_matrix(i, j) *
              real(get(one_minus_y).data()[i * number_of_angular_points]);
        }
      }
    }

    // zero out the lower left and upper right part of the matrix
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      for (size_t j = 0; j < number_of_radial_points; ++j) {
        operator_matrix(i + number_of_radial_points, j) = 0.0;
        operator_matrix(i, j + number_of_radial_points) = 0.0;
      }
    }

    // gather the contributions to the matrix blocks from the linear factors
    // each, we zero the first row
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      const size_t linear_factor_index = offset + i * number_of_angular_points;
      // upper left
      operator_matrix(i, i) +=
          real(get(linear_factor).data()[linear_factor_index] +
               get(linear_factor_of_conjugate).data()[linear_factor_index]);
      operator_matrix(0, i) = 0.0;
      // upper right
      operator_matrix(i, number_of_radial_points + i) -=
          imag(get(linear_factor).data()[linear_factor_index] -
               get(linear_factor_of_conjugate).data()[linear_factor_index]);
      operator_matrix(0, number_of_radial_points + i) = 0.0;
      // lower left
      operator_matrix(number_of_radial_points + i, i) +=
          imag(get(linear_factor).data()[linear_factor_index] +
               get(linear_factor_of_conjugate).data()[linear_factor_index]);
      operator_matrix(number_of_radial_points, i) = 0.0;
      // lower right
      operator_matrix(number_of_radial_points + i,
                      number_of_radial_points + i) +=
          real(get(linear_factor).data()[linear_factor_index] -
               get(linear_factor_of_conjugate).data()[linear_factor_index]);
      operator_matrix(number_of_radial_points, number_of_radial_points + i) =
          0.0;
    }
    operator_matrix(0, 0) = 1.0;
    operator_matrix(number_of_radial_points, number_of_radial_points) = 1.0;
    // put the data currently in integrand into a real DataVector of twice the
    // length
    linear_solve_buffer[offset * 2 * number_of_radial_points] =
        real(get(boundary).data()[offset]);
    linear_solve_buffer[(offset * 2 + 1) * number_of_radial_points] =
        imag(get(boundary).data()[offset]);
    DataVector linear_solve_buffer_view{
        linear_solve_buffer.data() + offset * 2 * number_of_radial_points,
        2 * number_of_radial_points};
    lapack::general_matrix_linear_solve(
        make_not_null(&linear_solve_buffer_view),
        make_not_null(&operator_matrix));
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  raw_transpose(make_not_null(reinterpret_cast<double*>(
                    get(*integral_result).data().data())),
                linear_solve_buffer.data(), number_of_radial_points,
                2 * number_of_angular_points);
}

template struct RadialIntegrateBondi<Tags::BoundaryValue, Tags::BondiBeta>;
template struct RadialIntegrateBondi<Tags::BoundaryValue, Tags::BondiQ>;
template struct RadialIntegrateBondi<Tags::BoundaryValue, Tags::BondiU>;
template struct RadialIntegrateBondi<Tags::BoundaryValue, Tags::BondiW>;
template struct RadialIntegrateBondi<Tags::BoundaryValue, Tags::BondiH>;
template struct RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue,
                                     Tags::BondiBeta>;
template struct RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue,
                                     Tags::BondiQ>;
template struct RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue,
                                     Tags::BondiU>;
template struct RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue,
                                     Tags::BondiW>;
template struct RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue,
                                     Tags::BondiH>;

}  // namespace Cce
