// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/IndefiniteIntegral.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
void indefinite_integral_impl(const gsl::not_null<DataVector*> integral,
                              const DataVector& integrand,
                              const Mesh<Dim>& mesh,
                              const size_t dim) noexcept {
  if (UNLIKELY(integral->size() != integrand.size())) {
    *integral = DataVector(integrand.size());
  }
  const Mesh<1> mesh_1d{mesh.extents(dim), gsl::at(mesh.basis(), dim),
                        gsl::at(mesh.quadrature(), dim)};
  const size_t num_pts = mesh_1d.number_of_grid_points();
  const Matrix& indefinite_integration_matrix =
      Spectral::integration_matrix(mesh_1d);
  for (StripeIterator stripe_it(mesh.extents(), dim); stripe_it; ++stripe_it) {
    dgemv_('N', num_pts, num_pts, 1.0, indefinite_integration_matrix.data(),
           num_pts, integrand.data() + stripe_it.offset(),  // NOLINT
           stripe_it.stride(), 0.0,
           integral->data() + stripe_it.offset(),  // NOLINT
           stripe_it.stride());
  }
}
}  // namespace

template <size_t Dim>
void indefinite_integral(const gsl::not_null<DataVector*> integral,
                         const DataVector& integrand, const Mesh<Dim>& mesh,
                         const size_t dim_to_integrate) noexcept {
  if (UNLIKELY(integral->size() != integrand.size())) {
    *integral = DataVector(integrand.size());
  }
  indefinite_integral_impl(integral, integrand, mesh, dim_to_integrate);
}

template <size_t Dim>
DataVector indefinite_integral(const DataVector& integrand,
                               const Mesh<Dim>& mesh,
                               const size_t dim_to_integrate) noexcept {
  DataVector integral{};
  indefinite_integral(&integral, integrand, mesh, dim_to_integrate);
  return integral;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)             \
  template DataVector indefinite_integral( \
      const DataVector&, const Mesh<GET_DIM(data)>&, const size_t) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
