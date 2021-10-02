// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/IndefiniteIntegral.hpp"

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
const Matrix empty_matrix = Matrix{};

template <size_t Dim, size_t... Indices>
auto make_integration_matrices(const Mesh<Dim>& mesh,
                               const size_t dim_to_integrate,
                               std::index_sequence<Indices...> /*meta*/) {
  return std::array<std::reference_wrapper<const Matrix>, Dim>{
      {(Indices == dim_to_integrate
            ? Spectral::integration_matrix(mesh.slice_through(dim_to_integrate))
            : empty_matrix)...}};
}
}  // namespace

template <size_t Dim, typename VectorType>
void indefinite_integral(const gsl::not_null<VectorType*> integral,
                         const VectorType& integrand, const Mesh<Dim>& mesh,
                         const size_t dim_to_integrate) {
  integral->destructive_resize(integrand.size());
  apply_matrices(integral,
                 make_integration_matrices<Dim>(
                     mesh, dim_to_integrate, std::make_index_sequence<Dim>{}),
                 integrand, mesh.extents());
}

template <size_t Dim, typename VectorType>
VectorType indefinite_integral(const VectorType& integrand,
                               const Mesh<Dim>& mesh,
                               const size_t dim_to_integrate) {
  VectorType integral{integrand.size()};
  indefinite_integral(make_not_null(&integral), integrand, mesh,
                      dim_to_integrate);
  return integral;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_VECTORTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                                \
  template GET_VECTORTYPE(data) indefinite_integral(                          \
      const GET_VECTORTYPE(data)&, const Mesh<GET_DIM(data)>&, const size_t); \
  template void indefinite_integral(                                          \
      const gsl::not_null<GET_VECTORTYPE(data)*> integral,                    \
      const GET_VECTORTYPE(data)&, const Mesh<GET_DIM(data)>&, const size_t);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (DataVector, ComplexDataVector))

#undef GET_DIM
#undef GET_VECTORTYPE
#undef INSTANTIATION
