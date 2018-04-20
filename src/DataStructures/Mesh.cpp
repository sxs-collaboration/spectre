// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Mesh.hpp"

#include <pup.h>  // IWYU pragma: keep

#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"

template <size_t Dim>
// clang-tidy: incorrectly reported redundancy in template expression
template <size_t N, Requires<(N > 0 and N == Dim)>>  // NOLINT
Mesh<Dim - 1> Mesh<Dim>::slice_away(const size_t d) const noexcept {
  std::array<size_t, Dim - 1> slice_extents{};
  std::array<Spectral::Basis, Dim - 1> slice_bases{};
  std::array<Spectral::Quadrature, Dim - 1> slice_quadratures{};
  for (size_t i = 0; i < Dim; ++i) {
    if (i < d) {
      gsl::at(slice_extents, i) = gsl::at(extents_.indices(), i);
      gsl::at(slice_bases, i) = gsl::at(bases_, i);
      gsl::at(slice_quadratures, i) = gsl::at(quadratures_, i);
    } else if (i > d) {
      gsl::at(slice_extents, i - 1) = gsl::at(extents_.indices(), i);
      gsl::at(slice_bases, i - 1) = gsl::at(bases_, i);
      gsl::at(slice_quadratures, i - 1) = gsl::at(quadratures_, i);
    }
  }
  return Mesh<Dim - 1>(slice_extents, slice_bases, slice_quadratures);
}

/// \cond HIDDEN_SYMBOLS
template <size_t Dim>
void Mesh<Dim>::pup(PUP::er& p) noexcept {
  p | extents_;
  p | bases_;
  p | quadratures_;
}

template <size_t Dim>
bool operator==(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs) noexcept {
  return lhs.extents() == rhs.extents() and lhs.basis() == rhs.basis() and
         lhs.quadrature() == rhs.quadrature();
}

template <size_t Dim>
bool operator!=(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GEN_OP(op, dim)                           \
  template bool operator op(const Mesh<dim>& lhs, \
                            const Mesh<dim>& rhs) noexcept;
#define INSTANTIATE_MESH(_, data) \
  template class Mesh<DIM(data)>; \
  GEN_OP(==, DIM(data))           \
  GEN_OP(!=, DIM(data))
#define INSTANTIATE_SLICE_AWAY(_, data)                                  \
  template Mesh<DIM(data) - 1> Mesh<DIM(data)>::slice_away(const size_t) \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_MESH, (0, 1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE_SLICE_AWAY, (1, 2, 3))

#undef DIM
#undef GEN_OP
#undef INSTANTIATE_MESH
#undef INSTANTIATE_SLICE_AWAY
/// \endcond
