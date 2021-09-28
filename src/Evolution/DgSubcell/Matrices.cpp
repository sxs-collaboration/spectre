// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Matrices.hpp"

#include <array>
#include <ostream>
#include <utility>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"  // IWYU pragma: keep

namespace evolution::dg::subcell::fd {
template <Spectral::Quadrature QuadratureType, size_t NumDgGridPoints1d,
          size_t Dim>
Matrix projection_matrix_cache_impl_helper(const Index<Dim>& subcell_extents) {
  // We currently require all dimensions to have the same number of grid
  // points, but this is checked in the calling function.
  const Index<Dim> dg_extents{NumDgGridPoints1d};
  const size_t num_subcells = subcell_extents.product();
  const size_t num_pts = dg_extents.product();

  const size_t num_rows = num_subcells;
  const size_t num_columns = num_pts;
  Matrix proj_matrix(num_rows, num_columns);

  std::array<Matrix, Dim> interpolation_matrices{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(interpolation_matrices, d) =
        Spectral::interpolation_matrix<Spectral::Basis::Legendre,
                                       QuadratureType>(
            dg_extents[d],
            Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                         Spectral::Quadrature::CellCentered>(
                subcell_extents[d]));
  }

  for (IndexIterator<Dim> subcell_it(subcell_extents); subcell_it;
       ++subcell_it) {
    for (IndexIterator<Dim> basis_it(dg_extents); basis_it; ++basis_it) {
      proj_matrix(subcell_it.collapsed_index(), basis_it.collapsed_index()) =
          interpolation_matrices[0](subcell_it()[0], basis_it()[0]);
      for (size_t d = 1; d < Dim; ++d) {
        proj_matrix(subcell_it.collapsed_index(), basis_it.collapsed_index()) *=
            gsl::at(interpolation_matrices, d)(subcell_it()[d], basis_it()[d]);
      }
    }
  }

  return proj_matrix;
}

template <Spectral::Quadrature QuadratureType, size_t NumDgGridPoints,
          size_t Dim>
const Matrix& projection_matrix_cache_impl(const Index<Dim>& subcell_extents) {
  static const Matrix result =
      projection_matrix_cache_impl_helper<QuadratureType, NumDgGridPoints>(
          subcell_extents);
  return result;
}

template <Spectral::Quadrature QuadratureType, size_t... Is, size_t Dim>
const Matrix& projection_matrix_impl(
    const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
    std::index_sequence<Is...> /*num_dg_grid_points*/) {
  ASSERT(
      dg_mesh.extents() == Index<Dim>(dg_mesh.extents(0)),
      "The mesh must have the same extents in all directions but has extents "
          << dg_mesh.extents());
  static const std::array<const Matrix& (*)(const Index<Dim>&), sizeof...(Is)>
      cache{{&projection_matrix_cache_impl<QuadratureType, Is>...}};
  return gsl::at(cache, dg_mesh.extents(0))(subcell_extents);
}

template <size_t Dim>
const Matrix& projection_matrix(const Mesh<Dim>& dg_mesh,
                                const Index<Dim>& subcell_extents) {
  ASSERT(dg_mesh.basis(0) == Spectral::Basis::Legendre,
         "FD Subcell projection only supports Legendre basis right now but got "
         "basis "
             << dg_mesh.basis(0));
  ASSERT(dg_mesh == Mesh<Dim>(dg_mesh.extents(0), dg_mesh.basis(0),
                              dg_mesh.quadrature(0)),
         "The mesh must be uniform but is " << dg_mesh);
  ASSERT(subcell_extents == Index<Dim>(subcell_extents[0]),
         "The subcell mesh must be uniform but is " << subcell_extents);
  switch (dg_mesh.quadrature(0)) {
    case Spectral::Quadrature::GaussLobatto:
      return projection_matrix_impl<Spectral::Quadrature::GaussLobatto>(
          dg_mesh, subcell_extents,
          std::make_index_sequence<
              Spectral::maximum_number_of_points<Spectral::Basis::Legendre> +
              1>{});
    case Spectral::Quadrature::Gauss:
      return projection_matrix_impl<Spectral::Quadrature::Gauss>(
          dg_mesh, subcell_extents,
          std::make_index_sequence<
              Spectral::maximum_number_of_points<Spectral::Basis::Legendre> +
              1>{});
    default:
      ERROR("Unsupported quadrature type in FD subcell projection matrix");
  };
}

double get_sixth_order_integration_coefficient(const size_t num_pts,
                                               const size_t index) {
  if (num_pts == 3) {
    return gsl::at(std::array<double, 3>{{9. / 8., 3. / 4., 9. / 8.}}, index);
  } else if (num_pts == 4) {
    return gsl::at(
        std::array<double, 4>{{13. / 12., 11. / 12., 11. / 12., 13. / 12.}},
        index);
  } else if (num_pts == 5) {
    return gsl::at(
        std::array<double, 5>{{1375. / 1152., 125. / 288., 335. / 192.,
                               125. / 288., 1375. / 1152.}},
        index);
  } else if (num_pts == 6) {
    return gsl::at(
        std::array<double, 6>{{741. / 640., 417. / 640., 381. / 320.,
                               381. / 320., 417. / 640., 741. / 640.}},
        index);
  } else if (num_pts == 7) {
    return gsl::at(
        std::array<double, 7>{{741. / 640., 3547. / 5760., 8111. / 5760.,
                               611. / 960., 8111. / 5760., 3547. / 5760.,
                               741. / 640.}},
        index);
  } else if (num_pts == 8) {
    return gsl::at(
        std::array<double, 8>{{1663. / 1440., 227. / 360., 323. / 240.,
                               139. / 160., 139. / 160., 323. / 240.,
                               227. / 360., 1663. / 1440.}},
        index);
  } else if (num_pts == 9) {
    return gsl::at(
        std::array<double, 9>{{1663. / 1440., 227. / 360., 1547. / 1152.,
                               245. / 288., 3001. / 2880., 245. / 288.,
                               1547. / 1152., 227. / 360., 1663. / 1440.}},
        index);
  } else if (num_pts == 10) {
    return gsl::at(
        std::array<double, 10>{{1375. / 1152., 125. / 288., 335. / 192.,
                                125. / 288., 1375. / 1152., 1375. / 1152.,
                                125. / 288., 335. / 192., 125. / 288.,
                                1375. / 1152.}},
        index);
  } else if (num_pts == 11) {
    return gsl::at(
        std::array<double, 11>{{1375. / 1152., 125. / 288., 335. / 192.,
                                2483. / 5760., 7183. / 5760., 863. / 960.,
                                7183. / 5760., 2483. / 5760., 335. / 192.,
                                125. / 288., 1375. / 1152.}},
        index);
  } else if (num_pts == 12) {
    return gsl::at(
        std::array<double, 12>{{1375. / 1152., 125. / 288., 335. / 192.,
                                2483. / 5760., 3583. / 2880., 2743. / 2880.,
                                2743. / 2880., 3583. / 2880., 2483. / 5760.,
                                335. / 192., 125. / 288., 1375. / 1152.}},
        index);
  } else if (num_pts == 13) {
    return gsl::at(
        std::array<double, 13>{{1375. / 1152., 125. / 288., 335. / 192.,
                                2483. / 5760., 3583. / 2880., 1823. / 1920.,
                                2897. / 2880., 1823. / 1920., 3583. / 2880.,
                                2483. / 5760., 335. / 192., 125. / 288.,
                                1375. / 1152.}},
        index);
  } else if (num_pts == 14) {
    return gsl::at(
        std::array<double, 14>{{1375. / 1152., 125. / 288., 335. / 192.,
                                2483. / 5760., 3583. / 2880., 1823. / 1920.,
                                5777. / 5760., 5777. / 5760., 1823. / 1920.,
                                3583. / 2880., 2483. / 5760., 335. / 192.,
                                125. / 288., 1375. / 1152.}},
        index);
  } else if (num_pts >= 15) {
    return gsl::at(
        std::array<double, 8>{{1375. / 1152., 125. / 288., 335. / 192.,
                               2483. / 5760., 3583. / 2880., 1823. / 1920.,
                               5777. / 5760., 1.}},

        index <= 7 ? index : index >= num_pts - 8 ? num_pts - 1 - index : 7);
  }
  ERROR("Cannot get coefficients for a mesh with only '"
        << num_pts << "' points. We need at least 5 points.");
}

// Function to get the 1d integration coefficient from the 1, 2, or 3d spatial
// location. The 1d coefficient is a product of the 3 coefficients in each
// direction.
template <size_t Dim>
double integration_weight(const Index<Dim>& extents, const Index<Dim>& index) {
  double result = get_sixth_order_integration_coefficient(extents[0], index[0]);
  for (size_t d = 1; d < Dim; ++d) {
    result *= get_sixth_order_integration_coefficient(extents[d], index[d]);
  }
  return result;
}

template <Spectral::Quadrature QuadratureType, size_t NumDgGridPoints1d,
          size_t Dim>
Matrix reconstruction_matrix_cache_impl_helper(
    const Index<Dim>& subcell_extents) {
  // We currently require all dimensions to have the same number of grid
  // points.
  const Index<Dim> dg_extents{NumDgGridPoints1d};
  const Matrix& proj_matrix =
      projection_matrix_cache_impl<QuadratureType, NumDgGridPoints1d>(
          subcell_extents);
  const size_t num_pts = dg_extents.product();
  const size_t num_subcells = subcell_extents.product();

  const size_t reconstruction_rows_and_cols = num_pts + 1;
  Matrix lhs_recons_matrix(reconstruction_rows_and_cols,
                           reconstruction_rows_and_cols, 0.0);
  // We use rhs_recons_matrix here as a temp buffer, we will fill it later.
  Matrix rhs_recons_matrix(num_pts + 1, num_subcells);
  dgemm_<true>('T', 'N', proj_matrix.columns(), proj_matrix.columns(),
               proj_matrix.rows(), 2.0, proj_matrix.data(), proj_matrix.rows(),
               proj_matrix.data(), proj_matrix.rows(), 0.0,
               rhs_recons_matrix.data(), proj_matrix.columns());

  for (size_t l = 0; l < num_pts; ++l) {
    for (size_t j = 0; j < num_pts; ++j) {
      lhs_recons_matrix(l, j) = *(rhs_recons_matrix.data() + (j + l * num_pts));
    }
  }

  std::array<const DataVector*, Dim> weights{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(weights, d) =
        &Spectral::quadrature_weights<Spectral::Basis::Legendre,
                                      QuadratureType>(dg_extents[d]);
  }
  for (IndexIterator<Dim> dg_it(dg_extents); dg_it; ++dg_it) {
    lhs_recons_matrix(dg_it.collapsed_index(), num_pts) =
        -(*weights[0])[dg_it()[0]];
    lhs_recons_matrix(num_pts, dg_it.collapsed_index()) =
        (*weights[0])[dg_it()[0]];
    for (size_t i = 1; i < Dim; ++i) {
      lhs_recons_matrix(dg_it.collapsed_index(), num_pts) *=
          (*gsl::at(weights, i))[dg_it()[i]];
      lhs_recons_matrix(num_pts, dg_it.collapsed_index()) *=
          (*gsl::at(weights, i))[dg_it()[i]];
    }
  }

  for (size_t k = 0; k < num_subcells; ++k) {
    for (size_t l = 0; l < num_pts; ++l) {
      rhs_recons_matrix(l, k) = 2.0 * proj_matrix(k, l);
    }
  }

  double deltas = 2.0 / subcell_extents[0];
  for (size_t d = 1; d < Dim; ++d) {
    deltas *= 2.0 / subcell_extents[d];
  }
  for (size_t i = 0; i < num_subcells; ++i) {
    rhs_recons_matrix(num_pts, i) = deltas;
  }
  for (IndexIterator<Dim> it(subcell_extents); it; ++it) {
    rhs_recons_matrix(num_pts, it.collapsed_index()) *=
        integration_weight(subcell_extents, *it);
  }

  const Matrix inv_lhs_recons_matrix = inv(lhs_recons_matrix);
  Matrix full_recons_matrix(inv_lhs_recons_matrix.rows(),
                            rhs_recons_matrix.columns());
  // Do matrix multipy with dgemm_ directly because that seems to be faster
  // than Blaze.
  dgemm_<true>('N', 'N', inv_lhs_recons_matrix.rows(),
               rhs_recons_matrix.columns(), inv_lhs_recons_matrix.columns(),
               1.0, inv_lhs_recons_matrix.data(), inv_lhs_recons_matrix.rows(),
               rhs_recons_matrix.data(), inv_lhs_recons_matrix.columns(), 0.0,
               full_recons_matrix.data(), inv_lhs_recons_matrix.rows());
  // exclude bottom row for Lagrange multiplier.
  Matrix reduced_recons_matrix(num_pts, num_subcells);
  for (size_t i = 0; i < num_pts; ++i) {
    for (size_t j = 0; j < num_subcells; ++j) {
      reduced_recons_matrix(i, j) = full_recons_matrix(i, j);
    }
  }

  return reduced_recons_matrix;
}

template <Spectral::Quadrature QuadratureType, size_t NumDgGridPoints,
          size_t Dim>
const Matrix& reconstruction_matrix_cache_impl(
    const Index<Dim>& subcell_extents) {
  static const Matrix result =
      reconstruction_matrix_cache_impl_helper<QuadratureType, NumDgGridPoints>(
          subcell_extents);
  return result;
}

template <Spectral::Quadrature QuadratureType, size_t... Is, size_t Dim>
const Matrix& reconstruction_matrix_impl(
    const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
    std::index_sequence<Is...> /*num_dg_grid_points*/) {
  static const std::array<const Matrix& (*)(const Index<Dim>&), sizeof...(Is)>
      cache{{&reconstruction_matrix_cache_impl<QuadratureType, Is, Dim>...}};
  return gsl::at(cache, dg_mesh.extents(0))(subcell_extents);
}

template <size_t Dim>
const Matrix& reconstruction_matrix(const Mesh<Dim>& dg_mesh,
                                    const Index<Dim>& subcell_extents) {
  ASSERT(dg_mesh.basis(0) == Spectral::Basis::Legendre,
         "FD Subcell reconstruction only supports Legendre basis right now.");
  ASSERT(dg_mesh == Mesh<Dim>(dg_mesh.extents(0), dg_mesh.basis(0),
                              dg_mesh.quadrature(0)),
         "The mesh must be uniform but is " << dg_mesh);
  ASSERT(subcell_extents == Index<Dim>(subcell_extents[0]),
         "The subcell mesh must be uniform but is " << subcell_extents);
  switch (dg_mesh.quadrature(0)) {
    case Spectral::Quadrature::GaussLobatto:
      return reconstruction_matrix_impl<Spectral::Quadrature::GaussLobatto>(
          dg_mesh, subcell_extents,
          std::make_index_sequence<
              Spectral::maximum_number_of_points<Spectral::Basis::Legendre> +
              1>{});
    case Spectral::Quadrature::Gauss:
      return reconstruction_matrix_impl<Spectral::Quadrature::Gauss>(
          dg_mesh, subcell_extents,
          std::make_index_sequence<
              Spectral::maximum_number_of_points<Spectral::Basis::Legendre> +
              1>{});
    default:
      ERROR("Unsupported quadrature type in FD subcell reconstruction matrix");
  };
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                             \
  template const Matrix& projection_matrix(const Mesh<GET_DIM(data)>&,     \
                                           const Index<GET_DIM(data)>&);   \
  template const Matrix& reconstruction_matrix(const Mesh<GET_DIM(data)>&, \
                                               const Index<GET_DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace evolution::dg::subcell::fd
