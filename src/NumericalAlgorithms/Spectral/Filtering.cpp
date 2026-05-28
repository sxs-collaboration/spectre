// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Filtering.hpp"

#include <cmath>

#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral::filtering {
Matrix exponential_filter(const Mesh<1>& mesh, const double alpha,
                          const unsigned half_power, const Parity parity) {
  if (UNLIKELY(mesh.number_of_grid_points() == 1)) {
    return Matrix(1, 1, 1.0);
  }
  if (mesh.basis(0) == Spectral::Basis::ZernikeB1) {
    ASSERT(parity != Parity::Uninitialized,
           "Need parity to be set to filter ZernikeB1");
    const size_t m = parity == Parity::Even ? 0 : 1;
    const size_t n = mesh.number_of_grid_points();
    const Matrix& nodal_to_modal =
        Spectral::nodal_to_modal_matrix<Spectral::Basis::ZernikeB1,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n, m, 2 * n - 2);
    const Matrix& modal_to_nodal =
        Spectral::modal_to_nodal_matrix<Spectral::Basis::ZernikeB1,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n, m, 2 * n - 2);
    Matrix filter_matrix(n - m, n - m, 0.0);
    const size_t order = n - 1 - m;
    filter_matrix(0, 0) = 1.0;
    for (size_t i = 1; i <= order; ++i) {
      filter_matrix(i, i) =
          exp(-alpha * pow(static_cast<double>(i) / static_cast<double>(order),
                           2 * half_power));
    }
    return modal_to_nodal * filter_matrix * nodal_to_modal;
  }
  const Matrix& nodal_to_modal = Spectral::nodal_to_modal_matrix(mesh);
  const Matrix& modal_to_nodal = Spectral::modal_to_nodal_matrix(mesh);
  Matrix filter_matrix(mesh.number_of_grid_points(),
                       mesh.number_of_grid_points(), 0.0);
  const double order = mesh.number_of_grid_points() - 1.0;
  for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
    filter_matrix(i, i) = exp(-alpha * pow(i / order, 2 * half_power));
  }
  return modal_to_nodal * filter_matrix * nodal_to_modal;
}

namespace {
template <Spectral::Basis BasisType, Spectral::Quadrature QuadratureType>
struct ZeroLowestModesImpl {
  Matrix operator()(const size_t num_points,
                    const size_t num_modes_to_zero) const {
    const Matrix& nodal_to_modal =
        Spectral::nodal_to_modal_matrix<BasisType, QuadratureType>(num_points);
    const Matrix& modal_to_nodal =
        Spectral::modal_to_nodal_matrix<BasisType, QuadratureType>(num_points);
    Matrix filter_matrix(num_points, num_points, 0.0);
    for (size_t i = num_modes_to_zero; i < num_points; ++i) {
      filter_matrix(i, i) = 1.0;
    }
    return Matrix(modal_to_nodal * filter_matrix * nodal_to_modal);
  }
};

struct ZernikeB1ZeroLowestModesImpl {
  Matrix operator()(const size_t n_dg, const size_t num_modes_to_zero,
                    const size_t m) const {
    const Matrix& nodal_to_modal =
        Spectral::nodal_to_modal_matrix<Spectral::Basis::ZernikeB1,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n_dg, m, 2 * n_dg - 2);
    const Matrix& modal_to_nodal =
        Spectral::modal_to_nodal_matrix<Spectral::Basis::ZernikeB1,
                                        Spectral::Quadrature::GaussRadauUpper>(
            n_dg, m, 2 * n_dg - 2);
    const size_t n_modes = n_dg - m;
    Matrix filter_matrix(n_modes, n_modes, 0.0);
    for (size_t i = num_modes_to_zero; i < n_modes; ++i) {
      filter_matrix(i, i) = 1.0;
    }
    return modal_to_nodal * filter_matrix * nodal_to_modal;
  }
};
}  // namespace

const Matrix& zero_lowest_modes(const Mesh<1>& mesh,
                                const size_t number_of_modes_to_zero,
                                const Parity parity) {
  ASSERT(number_of_modes_to_zero < mesh.extents(0),
         "For a 1d mesh with " << mesh.extents(0)
                               << " grid points, you cannot zero "
                               << number_of_modes_to_zero << " modes.");

  switch (mesh.basis(0)) {
    case Basis::Legendre:
      switch (mesh.quadrature(0)) {
        case Spectral::Quadrature::GaussLobatto: {
          constexpr size_t max_num_points =
              Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
          constexpr size_t min_num_points = Spectral::minimum_number_of_points<
              Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>;
          const auto cache =
              make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                                CacheRange<0_st, max_num_points>>(
                  ZeroLowestModesImpl<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
        }
        case Spectral::Quadrature::Gauss: {
          constexpr size_t max_num_points =
              Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
          constexpr size_t min_num_points =
              Spectral::minimum_number_of_points<Spectral::Basis::Legendre,
                                                 Spectral::Quadrature::Gauss>;
          const auto cache =
              make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                                CacheRange<0_st, max_num_points>>(
                  ZeroLowestModesImpl<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::Gauss>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
        }
        default:
          ERROR("Unsupported quadrature type in filtering lowest modes: "
                << mesh.quadrature(0));
      };
    case Basis::Chebyshev:
      switch (mesh.quadrature(0)) {
        case Spectral::Quadrature::GaussLobatto: {
          constexpr size_t max_num_points =
              Spectral::maximum_number_of_points<Spectral::Basis::Chebyshev>;
          constexpr size_t min_num_points = Spectral::minimum_number_of_points<
              Spectral::Basis::Chebyshev, Spectral::Quadrature::GaussLobatto>;
          const auto cache =
              make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                                CacheRange<0_st, max_num_points>>(
                  ZeroLowestModesImpl<Spectral::Basis::Chebyshev,
                                      Spectral::Quadrature::GaussLobatto>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
        }
        case Spectral::Quadrature::Gauss: {
          constexpr size_t max_num_points =
              Spectral::maximum_number_of_points<Spectral::Basis::Chebyshev>;
          constexpr size_t min_num_points =
              Spectral::minimum_number_of_points<Spectral::Basis::Chebyshev,
                                                 Spectral::Quadrature::Gauss>;
          const auto cache =
              make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                                CacheRange<0_st, max_num_points>>(
                  ZeroLowestModesImpl<Spectral::Basis::Chebyshev,
                                      Spectral::Quadrature::Gauss>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
        }
        default:
          ERROR("Unsupported quadrature type in filtering lowest modes: "
                << mesh.quadrature(0));
      };
    case Basis::ZernikeB1:
      switch (mesh.quadrature(0)) {
        case Spectral::Quadrature::GaussRadauUpper: {
          ASSERT(
              parity != Parity::Uninitialized,
              "Parity must be set to filter lowest modes on a ZernikeB1 mesh.");
          const size_t m = parity == Parity::Even ? 0 : 1;
          ASSERT(number_of_modes_to_zero < mesh.number_of_grid_points() - m,
                 "For ZernikeB1 with "
                     << mesh.number_of_grid_points()
                     << " grid points and parity " << (m == 0 ? "Even" : "Odd")
                     << " (" << mesh.number_of_grid_points() - m
                     << " modes), you cannot zero " << number_of_modes_to_zero
                     << " modes.");
          constexpr size_t max_num_pts =
              Spectral::maximum_number_of_points<Spectral::Basis::ZernikeB1>;
          constexpr size_t min_num_pts = Spectral::minimum_number_of_points<
              Spectral::Basis::ZernikeB1,
              Spectral::Quadrature::GaussRadauUpper>;
          const auto cache =
              make_static_cache<CacheRange<min_num_pts, max_num_pts + 1>,
                                CacheRange<0_st, max_num_pts>,
                                CacheRange<0_st, 2_st>>(
                  ZernikeB1ZeroLowestModesImpl{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero,
                       m);
        }
        default:
          ERROR("Unsupported quadrature type in filtering lowest modes: "
                << mesh.quadrature(0));
      };
    default:
      ERROR("Cannot filter basis type: " << mesh.basis(0));
  };
}
}  // namespace Spectral::filtering
