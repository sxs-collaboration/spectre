// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace dg {

namespace detail {
template <size_t Dim>
void apply_mass_matrix_impl(gsl::not_null<double*> data,
                            const Mesh<Dim>& mesh) noexcept;
}  // namespace detail

/*!
 * \brief Apply the DG mass matrix to the data, in the diagonal mass-matrix
 * approximation ("mass-lumping")
 *
 * The DG mass matrix is:
 *
 * \f{equation}
 * M_{pq} = \int_{\Omega_k} \psi_p(\xi) \psi_q(\xi) \mathrm{d}V
 * \f}
 *
 * where \f$\psi_p(\xi)\f$ are the basis functions on the element
 * \f$\Omega_k\f$. In the diagonal mass-matrix approximation ("mass-lumping") we
 * evaluate the integral directly on the collocation points, i.e. with a Gauss
 * or Gauss-Lobatto quadrature determined by the element mesh. Then it reduces
 * to:
 *
 * \f{equation}
 * M_{pq} \approx \delta_{pq} \prod_{i=1}^d w_{p_i}
 * \f}
 *
 * where \f$d\f$ is the spatial dimension and \f$w_{p_i}\f$ are the quadrature
 * weights in dimension \f$i\f$. To apply the mass matrix in coordinates
 * different than logical, or to account for a curved background metric, the
 * data can be pre-multiplied with the Jacobian determinant and/or a metric
 * determinant.
 *
 * \note The mass-lumping is exact on Legendre-Gauss meshes, but omits a
 * correction term on Legendre-Gauss-Lobatto meshes.
 */
/// @{
template <size_t Dim>
void apply_mass_matrix(const gsl::not_null<DataVector*> data,
                       const Mesh<Dim>& mesh) noexcept {
  ASSERT(data->size() == mesh.number_of_grid_points(),
         "The DataVector has size " << data->size() << ", but expected size "
                                    << mesh.number_of_grid_points()
                                    << " on the given mesh.");
  detail::apply_mass_matrix_impl(data->data(), mesh);
}

template <size_t Dim, typename TagsList>
void apply_mass_matrix(const gsl::not_null<Variables<TagsList>*> data,
                       const Mesh<Dim>& mesh) noexcept {
  const size_t num_points = data->number_of_grid_points();
  ASSERT(num_points == mesh.number_of_grid_points(),
         "The Variables data has "
             << num_points << " grid points, but expected "
             << mesh.number_of_grid_points() << " on the given mesh.");
  constexpr size_t num_comps =
      Variables<TagsList>::number_of_independent_components;
  for (size_t i = 0; i < num_comps; ++i) {
    detail::apply_mass_matrix_impl(data->data() + i * num_points, mesh);
  }
}
/// @}

}  // namespace dg
