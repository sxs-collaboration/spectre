// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Filtering.hpp"

#include <cmath>

#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral::filtering {
Matrix exponential_filter(const Mesh<1>& mesh, const double alpha,
                          const unsigned half_power) {
  if (UNLIKELY(mesh.number_of_grid_points() == 1)) {
    return Matrix(1, 1, 1.0);
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
template <SpatialDiscretization::Basis Basis,
          SpatialDiscretization::Quadrature Quadrature>
struct ZeroLowestModesImpl {
  Matrix operator()(const size_t num_points,
                    const size_t num_modes_to_zero) const {
    const Matrix& nodal_to_modal =
        Spectral::nodal_to_modal_matrix<Basis, Quadrature>(num_points);
    const Matrix& modal_to_nodal =
        Spectral::modal_to_nodal_matrix<Basis, Quadrature>(num_points);
    Matrix filter_matrix(num_points, num_points, 0.0);
    for (size_t i = num_modes_to_zero; i < num_points; ++i) {
      filter_matrix(i, i) = 1.0;
    }
    return Matrix(modal_to_nodal * filter_matrix * nodal_to_modal);
  }
};
}  // namespace

const Matrix& zero_lowest_modes(const Mesh<1>& mesh,
                                const size_t number_of_modes_to_zero) {
  ASSERT(number_of_modes_to_zero < mesh.extents(0),
         "For a 1d mesh with " << mesh.extents(0)
                               << " grid points, you cannot zero "
                               << number_of_modes_to_zero << " modes.");

  switch (mesh.basis(0)) {
    case SpatialDiscretization::Basis::Legendre:
      switch (mesh.quadrature(0)) {
        case SpatialDiscretization::Quadrature::GaussLobatto: {
          constexpr size_t max_num_points = Spectral::maximum_number_of_points<
              SpatialDiscretization::Basis::Legendre>;
          constexpr size_t min_num_points = Spectral::minimum_number_of_points<
              SpatialDiscretization::Basis::Legendre,
              SpatialDiscretization::Quadrature::GaussLobatto>;
          const auto cache =
              make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                                CacheRange<0_st, max_num_points>>(
                  ZeroLowestModesImpl<
                      SpatialDiscretization::Basis::Legendre,
                      SpatialDiscretization::Quadrature::GaussLobatto>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
        }
        case SpatialDiscretization::Quadrature::Gauss: {
          constexpr size_t max_num_points = Spectral::maximum_number_of_points<
              SpatialDiscretization::Basis::Legendre>;
          constexpr size_t min_num_points = Spectral::minimum_number_of_points<
              SpatialDiscretization::Basis::Legendre,
              SpatialDiscretization::Quadrature::Gauss>;
          const auto cache = make_static_cache<
              CacheRange<min_num_points, max_num_points + 1>,
              CacheRange<0_st, max_num_points>>(
              ZeroLowestModesImpl<SpatialDiscretization::Basis::Legendre,
                                  SpatialDiscretization::Quadrature::Gauss>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
        }
        default:
          ERROR("Unsupported quadrature type in filtering lowest modes: "
                << mesh.quadrature(0));
      };
    case SpatialDiscretization::Basis::Chebyshev:
      switch (mesh.quadrature(0)) {
        case SpatialDiscretization::Quadrature::GaussLobatto: {
          constexpr size_t max_num_points = Spectral::maximum_number_of_points<
              SpatialDiscretization::Basis::Chebyshev>;
          constexpr size_t min_num_points = Spectral::minimum_number_of_points<
              SpatialDiscretization::Basis::Chebyshev,
              SpatialDiscretization::Quadrature::GaussLobatto>;
          const auto cache =
              make_static_cache<CacheRange<min_num_points, max_num_points + 1>,
                                CacheRange<0_st, max_num_points>>(
                  ZeroLowestModesImpl<
                      SpatialDiscretization::Basis::Chebyshev,
                      SpatialDiscretization::Quadrature::GaussLobatto>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
        }
        case SpatialDiscretization::Quadrature::Gauss: {
          constexpr size_t max_num_points = Spectral::maximum_number_of_points<
              SpatialDiscretization::Basis::Chebyshev>;
          constexpr size_t min_num_points = Spectral::minimum_number_of_points<
              SpatialDiscretization::Basis::Chebyshev,
              SpatialDiscretization::Quadrature::Gauss>;
          const auto cache = make_static_cache<
              CacheRange<min_num_points, max_num_points + 1>,
              CacheRange<0_st, max_num_points>>(
              ZeroLowestModesImpl<SpatialDiscretization::Basis::Chebyshev,
                                  SpatialDiscretization::Quadrature::Gauss>{});
          return cache(mesh.number_of_grid_points(), number_of_modes_to_zero);
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
