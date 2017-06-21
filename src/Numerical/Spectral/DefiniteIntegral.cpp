
#include "Numerical/Spectral/DefiniteIntegral.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Numerical/Spectral/LegendreGaussLobatto.hpp"
#include "Utilities/Blas.hpp"

namespace Basis {
namespace lgl {
namespace detail {

template <size_t Dim>
DataVector integrate_over_last_dimension(const DataVector& f,
                                         const Index<Dim>& extents) {
  static_assert(Dim > 1, "Expect dimension to be at least 2.");
  const size_t extents_in_last_dim = extents[Dim - 1];
  const DataVector& w = Basis::lgl::quadrature_weights(extents_in_last_dim);
  const size_t reduced_size = extents.template product<Dim - 1>();
  DataVector integrated_data(reduced_size, 0.);
  dgemv_('N', reduced_size, extents_in_last_dim, 1., f.data(), reduced_size,
         w.data(), 1, 0., integrated_data.data(), 1);
  return integrated_data;
}
}  // namespace detail

template <size_t Dim>
double definite_integral(const DataVector& f, const Index<Dim>& extents) noexcept{
  ASSERT(f.size() == extents.product(),
         "size = " << f.size() << ", product = " << extents.product());
  return definite_integral(detail::integrate_over_last_dimension(f, extents),
                           extents.slice_away(Dim - 1));
}

template <>
double definite_integral<>(const DataVector& f, const Index<1>& extents) noexcept{
  const size_t N = f.size();
  ASSERT(N == extents.product(),
         "N = " << N << ", product = " << extents.product());
  const DataVector& w = Basis::lgl::quadrature_weights(N);
  return ddot_(N, w.data(), 1, f.data(), 1);
}

}  // namespace lgl
}  // namespace Basis

template double Basis::lgl::definite_integral<2>(const DataVector&,
                                                 const Index<2>&);
template double Basis::lgl::definite_integral<3>(const DataVector&,
                                                 const Index<3>&);
