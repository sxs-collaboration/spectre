// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Evolution/DgSubcell/ProjectionTestHelpers.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace TestHelpers::evolution::dg::subcell {
// computes a simple polynomial over the grid that we then project and
// reconstruct in the tests.
template <size_t Dim, typename Fr>
DataVector cell_values(const size_t max_polynomial_degree_plus_one,
                       const tnsr::I<DataVector, Dim, Fr>& coords) noexcept {
  DataVector result(get<0>(coords).size(), 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    for (size_t i = 0; i < max_polynomial_degree_plus_one; ++i) {
      result += pow(coords.get(d), i);
    }
  }
  return result;
}

// Computes the average in each finite volume cell multiplied by the cell's
// volume.
template <size_t Dim>
DataVector cell_averages_times_volume(
    const size_t max_polynomial_degree_plus_one,
    const Index<Dim>& subcell_extents) noexcept {
  Index<Dim> subcell_boundary_extents{};
  for (size_t d = 0; d < Dim; ++d) {
    subcell_boundary_extents[d] = subcell_extents[d] + 1;
  }
  std::array<DataVector, Dim> subcell_boundary_coords{};

  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(subcell_boundary_coords, d) =
        DataVector(subcell_boundary_extents.product());
    const auto& collocation_points_in_this_dim =
        Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                     Spectral::Quadrature::FaceCentered>(
            subcell_boundary_extents[d]);
    for (IndexIterator<Dim> index(subcell_boundary_extents); index; ++index) {
      gsl::at(subcell_boundary_coords, d)[index.collapsed_index()] =
          collocation_points_in_this_dim[index()[d]];
    }
  }

  DataVector integral_values_at_bounadry_points(
      subcell_boundary_extents.product(), 0.0);
  // Compute the integral pointwise over the subcell boundary points.
  for (size_t d = 0; d < Dim; ++d) {
    for (size_t i = 0; i < max_polynomial_degree_plus_one; ++i) {
      DataVector temp =
          pow(gsl::at(subcell_boundary_coords, d), i + 1) / (i + 1.0);
      for (size_t j = 0; j < Dim; ++j) {
        if (j != d) {
          temp *= gsl::at(subcell_boundary_coords, j);
        }
      }
      integral_values_at_bounadry_points += temp;
    }
  }
  // Compute the average
  DataVector result(subcell_extents.product());
  for (IndexIterator<Dim> subcell_it(subcell_extents); subcell_it;
       ++subcell_it) {
    if (Dim == 1) {
      result[subcell_it.collapsed_index()] =
          integral_values_at_bounadry_points[subcell_it()[0] + 1] -
          integral_values_at_bounadry_points[subcell_it()[0]];
    } else if (Dim == 2) {
      Index<Dim> uu_bound{};
      Index<Dim> ul_bound{};
      Index<Dim> lu_bound{};
      Index<Dim> ll_bound{};
      for (size_t d = 0; d < Dim; ++d) {
        uu_bound[d] = subcell_it()[d] + 1;
        ll_bound[d] = subcell_it()[d];
        ul_bound[d] = d == 0 ? subcell_it()[d] + 1 : subcell_it()[d];
        lu_bound[d] = d == 0 ? subcell_it()[d] : subcell_it()[d] + 1;
      }
      result[subcell_it.collapsed_index()] =
          integral_values_at_bounadry_points[collapsed_index(
              uu_bound, subcell_boundary_extents)] -
          integral_values_at_bounadry_points[collapsed_index(
              ul_bound, subcell_boundary_extents)] +
          integral_values_at_bounadry_points[collapsed_index(
              ll_bound, subcell_boundary_extents)] -
          integral_values_at_bounadry_points[collapsed_index(
              lu_bound, subcell_boundary_extents)];
    } else if (Dim == 3) {
      Index<Dim> uuu_bound{};
      Index<Dim> uul_bound{};
      Index<Dim> ulu_bound{};
      Index<Dim> luu_bound{};
      Index<Dim> ull_bound{};
      Index<Dim> lul_bound{};
      Index<Dim> llu_bound{};
      Index<Dim> lll_bound{};
      for (size_t d = 0; d < Dim; ++d) {
        uuu_bound[d] = subcell_it()[d] + 1;
        uul_bound[d] = d == 2 ? subcell_it()[d] : subcell_it()[d] + 1;
        ulu_bound[d] = d == 1 ? subcell_it()[d] : subcell_it()[d] + 1;
        luu_bound[d] = d == 0 ? subcell_it()[d] : subcell_it()[d] + 1;
        ull_bound[d] = d == 0 ? subcell_it()[d] + 1 : subcell_it()[d];
        lul_bound[d] = d == 1 ? subcell_it()[d] + 1 : subcell_it()[d];
        llu_bound[d] = d == 2 ? subcell_it()[d] + 1 : subcell_it()[d];
        lll_bound[d] = subcell_it()[d];
      }
      result[subcell_it.collapsed_index()] =
          integral_values_at_bounadry_points[collapsed_index(
              uuu_bound, subcell_boundary_extents)] -
          integral_values_at_bounadry_points[collapsed_index(
              luu_bound, subcell_boundary_extents)] -
          integral_values_at_bounadry_points[collapsed_index(
              ulu_bound, subcell_boundary_extents)] +
          integral_values_at_bounadry_points[collapsed_index(
              llu_bound, subcell_boundary_extents)] -
          integral_values_at_bounadry_points[collapsed_index(
              uul_bound, subcell_boundary_extents)] +
          integral_values_at_bounadry_points[collapsed_index(
              lul_bound, subcell_boundary_extents)] +
          integral_values_at_bounadry_points[collapsed_index(
              ull_bound, subcell_boundary_extents)] -
          integral_values_at_bounadry_points[collapsed_index(
              lll_bound, subcell_boundary_extents)];
    }
  }
  return result;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                         \
  template DataVector cell_values(                                     \
      size_t max_polynomial_degree_plus_one,                           \
      const tnsr::I<DataVector, GET_DIM(data), Frame::ElementLogical>& \
          coords) noexcept;                                            \
  template DataVector cell_values(                                     \
      size_t max_polynomial_degree_plus_one,                           \
      const tnsr::I<DataVector, GET_DIM(data), Frame::Inertial>&       \
          coords) noexcept;                                            \
  template DataVector cell_averages_times_volume(                      \
      size_t max_polynomial_degree_plus_one,                           \
      const Index<GET_DIM(data)>& subcell_extents) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION

}  // namespace TestHelpers::evolution::dg::subcell
