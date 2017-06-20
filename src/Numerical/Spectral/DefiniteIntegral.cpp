// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Numerical/Spectral/DefiniteIntegral.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Numerical/Spectral/LegendreGaussLobatto.hpp"
#include "Utilities/Blas.hpp"

namespace {

template <size_t Dim>
Scalar<DataVector> integrate_over_last_dimension(const DataVector& integrand,
                                                 const Index<Dim>& extents) {
  static_assert(Dim > 1, "Expect dimension to be at least 2.");
  const size_t extents_in_last_dim = extents[Dim - 1];
  const DataVector& weights =
      Basis::lgl::quadrature_weights(extents_in_last_dim);
  const size_t reduced_size = extents.template product<Dim - 1>();
  Scalar<DataVector> integrated_data(reduced_size, 0.);
  dgemv_('N', reduced_size, extents_in_last_dim, 1., integrand.data(),
         reduced_size, weights.data(), 1, 0., integrated_data.get().data(), 1);
  return integrated_data;
}
}  // namespace

namespace Basis {
namespace lgl {

template <size_t Dim>
double definite_integral(const Scalar<DataVector>& integrand,
                         const Index<Dim>& extents) noexcept {
  ASSERT(integrand.get().size() == extents.product(),
         "size = " << integrand.size() << ", product = " << extents.product());
  return definite_integral(
      integrate_over_last_dimension(integrand.get(), extents),
      extents.slice_away(Dim - 1));
}

template <>
double definite_integral<1>(const Scalar<DataVector>& integrand,
                            const Index<1>& extents) noexcept {
  const size_t num_points = integrand.begin()->size();
  ASSERT(num_points == extents.product(),
         "num_points = " << num_points << ", product = " << extents.product());
  const DataVector& weights = Basis::lgl::quadrature_weights(num_points);
  return ddot_(num_points, weights.data(), 1, integrand.get().data(), 1);
}

}  // namespace lgl
}  // namespace Basis

/// \cond
template double Basis::lgl::definite_integral<2>(const Scalar<DataVector>&,
                                                 const Index<2>&);
template double Basis::lgl::definite_integral<3>(const Scalar<DataVector>&,
                                                 const Index<3>&);
/// \endcond
