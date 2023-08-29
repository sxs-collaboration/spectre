// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"

#include <algorithm>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep
#include <pup_stl.h>

#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"

template <size_t Dim>
// clang-tidy: incorrectly reported redundancy in template expression
template <size_t N, Requires<(N > 0 and N == Dim)>>  // NOLINT
Mesh<Dim - 1> Mesh<Dim>::slice_away(const size_t d) const {
  ASSERT(d < Dim, "Tried to slice away non-existing dimension "
                      << d << " of " << Dim << "-dimensional mesh.");
  std::array<size_t, Dim - 1> dims{};
  for (size_t i = 0; i < d; i++) {
    gsl::at(dims, i) = i;
  }
  for (size_t i = d + 1; i < Dim; i++) {
    gsl::at(dims, i - 1) = i;
  }
  return slice_through(dims);
}

template <size_t Dim>
template <size_t SliceDim, Requires<(SliceDim <= Dim)>>
Mesh<SliceDim> Mesh<Dim>::slice_through(
    const std::array<size_t, SliceDim>& dims) const {
  // Check for duplicates in `dims`
  ASSERT(
      [&dims]() {
        auto sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end());
        auto last_unique = std::unique(sorted_dims.begin(), sorted_dims.end());
        return last_unique == sorted_dims.end();
      }(),
      "Dimensions to slice through contain duplicates.");
  std::array<size_t, SliceDim> slice_extents{};
  std::array<Spectral::Basis, SliceDim> slice_bases{};
  std::array<Spectral::Quadrature, SliceDim> slice_quadratures{};
  for (size_t i = 0; i < SliceDim; ++i) {
    const auto& d = gsl::at(dims, i);
    ASSERT(d < Dim, "Tried to slice through non-existing dimension "
                        << d << " of " << Dim << "-dimensional mesh.");
    gsl::at(slice_extents, i) = gsl::at(extents_.indices(), d);
    gsl::at(slice_bases, i) = gsl::at(bases_, gsl::at(dims, i));
    gsl::at(slice_quadratures, i) = gsl::at(quadratures_, gsl::at(dims, i));
  }
  return Mesh<SliceDim>(std::move(slice_extents), std::move(slice_bases),
                        std::move(slice_quadratures));
}

template <size_t Dim>
std::array<Mesh<1>, Dim> Mesh<Dim>::slices() const {
  std::array<Mesh<1>, Dim> result{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = Mesh<1>(extents(d), basis(d), quadrature(d));
  }
  return result;
}

template <size_t Dim>
void Mesh<Dim>::pup(PUP::er& p) {
  p | extents_;
  p | bases_;
  p | quadratures_;
}

template <size_t Dim>
bool is_isotropic(const Mesh<Dim>& mesh) {
  if constexpr (Dim == 0 or Dim == 1) {
    return true;
  } else {
    return mesh.extents() == Index<Dim>(mesh.extents(0)) and
           mesh.basis() == make_array<Dim>(mesh.basis(0)) and
           mesh.quadrature() == make_array<Dim>(mesh.quadrature(0));
  }
}

template <size_t Dim>
bool operator==(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs) {
  return lhs.extents() == rhs.extents() and lhs.basis() == rhs.basis() and
         lhs.quadrature() == rhs.quadrature();
}

template <size_t Dim>
bool operator!=(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const Mesh<Dim>& mesh) {
  using ::operator<<;
  return os << '[' << mesh.extents() << ',' << mesh.basis() << ','
            << mesh.quadrature() << ']';
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GEN_OP(op, dim) \
  template bool operator op(const Mesh<dim>& lhs, const Mesh<dim>& rhs);
#define INSTANTIATE_MESH(_, data)                                 \
  template class Mesh<DIM(data)>;                                 \
  GEN_OP(==, DIM(data))                                           \
  GEN_OP(!=, DIM(data))                                           \
  template std::ostream& operator<<(std::ostream& os,             \
                                    const Mesh<DIM(data)>& mesh); \
  template bool is_isotropic(const Mesh<DIM(data)>& mesh);

#define INSTANTIATE_SLICE_AWAY(_, data) \
  template Mesh<DIM(data) - 1> Mesh<DIM(data)>::slice_away(const size_t) const;
template Mesh<0> Mesh<0>::slice_through(const std::array<size_t, 0>&) const;
template Mesh<0> Mesh<1>::slice_through(const std::array<size_t, 0>&) const;
template Mesh<1> Mesh<1>::slice_through(const std::array<size_t, 1>&) const;
template Mesh<0> Mesh<2>::slice_through(const std::array<size_t, 0>&) const;
template Mesh<1> Mesh<2>::slice_through(const std::array<size_t, 1>&) const;
template Mesh<2> Mesh<2>::slice_through(const std::array<size_t, 2>&) const;
template Mesh<0> Mesh<3>::slice_through(const std::array<size_t, 0>&) const;
template Mesh<1> Mesh<3>::slice_through(const std::array<size_t, 1>&) const;
template Mesh<2> Mesh<3>::slice_through(const std::array<size_t, 2>&) const;
template Mesh<3> Mesh<3>::slice_through(const std::array<size_t, 3>&) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_MESH, (0, 1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE_SLICE_AWAY, (1, 2, 3))

#undef DIM
#undef GEN_OP
#undef INSTANTIATE_MESH
#undef INSTANTIATE_SLICE_AWAY
