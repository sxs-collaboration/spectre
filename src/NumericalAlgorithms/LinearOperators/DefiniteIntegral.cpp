// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"

#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"

namespace {

template <size_t Dim>
DataVector integrate_over_last_dimension(const DataVector& integrand,
                                         const Mesh<Dim>& mesh) noexcept {
  static_assert(Dim > 1, "Expect dimension to be at least 2.");
  const DataVector& weights =
      Spectral::quadrature_weights(mesh.slice_through(Dim - 1));
  const size_t reduced_size = mesh.slice_away(Dim - 1).number_of_grid_points();
  DataVector integrated_data(reduced_size, 0.);
  dgemv_('N', reduced_size, mesh.extents(Dim - 1), 1., integrand.data(),
         reduced_size, weights.data(), 1, 0., integrated_data.data(), 1);
  return integrated_data;
}
}  // namespace

template <size_t Dim>
double definite_integral(const DataVector& integrand,
                         const Mesh<Dim>& mesh) noexcept {
  const size_t num_grid_points = mesh.number_of_grid_points();
  ASSERT(integrand.size() == num_grid_points,
         "num_grid_points = " << num_grid_points
                              << ", integrand size = " << integrand.size());
  return definite_integral(integrate_over_last_dimension(integrand, mesh),
                           mesh.slice_away(Dim - 1));
}

template <>
double definite_integral<1>(const DataVector& integrand,
                            const Mesh<1>& mesh) noexcept {
  const size_t num_grid_points = mesh.number_of_grid_points();
  ASSERT(integrand.size() == num_grid_points,
         "num_grid_points = " << num_grid_points
                              << ", integrand size = " << integrand.size());
  const DataVector& weights = Spectral::quadrature_weights(mesh);
  return ddot_(num_grid_points, weights.data(), 1, integrand.data(), 1);
}

/// \cond
template double definite_integral<2>(const DataVector&,
                                     const Mesh<2>&) noexcept;
template double definite_integral<3>(const DataVector&,
                                     const Mesh<3>&) noexcept;
/// \endcond
