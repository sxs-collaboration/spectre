// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Mesh.hpp"

#include <ostream>
#include <pup.h>
#include <pup_stl.h>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
KOKKOS_FUNCTION uint8_t combine(const Spectral::Basis basis,
                                const Spectral::Quadrature quadrature) {
  return static_cast<uint8_t>(basis) bitor static_cast<uint8_t>(quadrature);
}

KOKKOS_FUNCTION Spectral::Basis extract_basis(const uint8_t bits) {
  return static_cast<Spectral::Basis>(0b11110000 bitand bits);
}

KOKKOS_FUNCTION Spectral::Quadrature extract_quadrature(const uint8_t bits) {
  return static_cast<Spectral::Quadrature>(0b00001111 bitand bits);
}
}  // namespace

template <size_t Dim>
KOKKOS_FUNCTION Mesh<Dim>::Mesh(const size_t isotropic_extents,
                                const Spectral::Basis basis,
                                const Spectral::Quadrature quadrature) {
  static_assert(sizeof(Mesh<Dim>) == (Dim == 0 ? 2 : 2 * Dim));
  if constexpr (Dim > 0) {
    ASSERT(isotropic_extents <= 255, "Cannot have more than 255 grid points");
    extents_[0] = isotropic_extents;
    quadrature_and_basis_[0] = combine(basis, quadrature);
    if constexpr (Dim > 1) {
      extents_[1] = isotropic_extents;
      quadrature_and_basis_[1] = combine(basis, quadrature);
      if constexpr (Dim > 2) {
        extents_[2] = isotropic_extents;
        quadrature_and_basis_[2] = combine(basis, quadrature);
      }
    }
  } else {
    // Silence compiler warnings
    (void)isotropic_extents;
    (void)basis;
    (void)quadrature;
  }
}

template <size_t Dim>
KOKKOS_FUNCTION Mesh<Dim>::Mesh(const std::array<size_t, Dim>& extents,
                                const Spectral::Basis basis,
                                const Spectral::Quadrature quadrature) {
  if constexpr (Dim > 0) {
    ASSERT(
        extents[0] <= 255,
        "Cannot have more than 255 grid points in direction 0: " << extents[0]);
    extents_[0] = extents[0];
    quadrature_and_basis_[0] = combine(basis, quadrature);
    if constexpr (Dim > 1) {
      ASSERT(extents[1] <= 255,
             "Cannot have more than 255 grid points in direction 1: "
                 << extents[1]);
      extents_[1] = extents[1];
      quadrature_and_basis_[1] = combine(basis, quadrature);
      if constexpr (Dim > 2) {
        extents_[2] = extents[2];
        ASSERT(extents[2] <= 255,
               "Cannot have more than 255 grid points in direction 2: "
                   << extents[2]);
        quadrature_and_basis_[2] = combine(basis, quadrature);
      }
    }
  } else {
    // Silence compiler warnings
    (void)extents;
    (void)basis;
    (void)quadrature;
  }
}

template <size_t Dim>
KOKKOS_FUNCTION Mesh<Dim>::Mesh(
    const std::array<size_t, Dim>& extents,
    const std::array<Spectral::Basis, Dim>& bases,
    const std::array<Spectral::Quadrature, Dim>& quadratures) {
  if constexpr (Dim > 0) {
    ASSERT(
        extents[0] <= 255,
        "Cannot have more than 255 grid points in direction 0: " << extents[0]);
    extents_[0] = extents[0];
    quadrature_and_basis_[0] = combine(bases[0], quadratures[0]);
    if constexpr (Dim > 1) {
      ASSERT(extents[1] <= 255,
             "Cannot have more than 255 grid points in direction 1: "
                 << extents[1]);
      extents_[1] = extents[1];
      quadrature_and_basis_[1] = combine(bases[1], quadratures[1]);
      if constexpr (Dim > 2) {
        ASSERT(extents[2] <= 255,
               "Cannot have more than 255 grid points in direction 2: "
                   << extents[2]);
        extents_[2] = extents[2];
        quadrature_and_basis_[2] = combine(bases[2], quadratures[2]);
      }
    }
  } else {
    // Silence compiler warnings
    (void)extents;
    (void)bases;
    (void)quadratures;
  }
}

template <size_t Dim>
KOKKOS_FUNCTION Index<Dim> Mesh<Dim>::extents() const {
  if constexpr (Dim == 0) {
    return Index<Dim>{};
  } else if constexpr (Dim == 1) {
    return Index<Dim>{static_cast<size_t>(extents_[0])};
  } else if constexpr (Dim == 2) {
    return Index<Dim>{static_cast<size_t>(extents_[0]),
                      static_cast<size_t>(extents_[1])};
  } else {
    return Index<Dim>{static_cast<size_t>(extents_[0]),
                      static_cast<size_t>(extents_[1]),
                      static_cast<size_t>(extents_[2])};
  }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif
template <size_t Dim>
KOKKOS_FUNCTION size_t Mesh<Dim>::extents(const size_t d) const {
  ASSERT(d < Dim,
         "Cannot get extent for dim " << d << " in a mesh of Dim " << Dim);
  return gsl::at(extents_, d);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <size_t Dim>
KOKKOS_FUNCTION size_t Mesh<Dim>::number_of_grid_points() const {
  if constexpr (Dim == 0) {
    return 1;
  } else if constexpr (Dim == 1) {
    return static_cast<size_t>(extents_[0]);
  } else if constexpr (Dim == 2) {
    // cast first so we don't overflow
    return static_cast<size_t>(extents_[0]) * static_cast<size_t>(extents_[1]);
  } else {
    // cast first so we don't overflow
    return static_cast<size_t>(extents_[0]) * static_cast<size_t>(extents_[1]) *
           static_cast<size_t>(extents_[2]);
  }
}

template <size_t Dim>
KOKKOS_FUNCTION size_t Mesh<Dim>::storage_index(const Index<Dim>& index) const {
  return collapsed_index(index, extents());
}

template <size_t Dim>
KOKKOS_FUNCTION std::array<Spectral::Basis, Dim> Mesh<Dim>::basis() const {
  if constexpr (Dim == 0) {
    return {};
  } else if constexpr (Dim == 1) {
    return {extract_basis(quadrature_and_basis_[0])};
  } else if constexpr (Dim == 2) {
    return {extract_basis(quadrature_and_basis_[0]),
            extract_basis(quadrature_and_basis_[1])};
  } else {
    return {extract_basis(quadrature_and_basis_[0]),
            extract_basis(quadrature_and_basis_[1]),
            extract_basis(quadrature_and_basis_[2])};
  }
}

template <size_t Dim>
KOKKOS_FUNCTION Spectral::Basis Mesh<Dim>::basis(const size_t d) const {
  return extract_basis(gsl::at(quadrature_and_basis_, d));
}

template <size_t Dim>
KOKKOS_FUNCTION std::array<Spectral::Quadrature, Dim> Mesh<Dim>::quadrature()
    const {
  if constexpr (Dim == 0) {
    return {};
  } else if constexpr (Dim == 1) {
    return {extract_quadrature(quadrature_and_basis_[0])};
  } else if constexpr (Dim == 2) {
    return {extract_quadrature(quadrature_and_basis_[0]),
            extract_quadrature(quadrature_and_basis_[1])};
  } else {
    return {extract_quadrature(quadrature_and_basis_[0]),
            extract_quadrature(quadrature_and_basis_[1]),
            extract_quadrature(quadrature_and_basis_[2])};
  }
}

template <size_t Dim>
KOKKOS_FUNCTION Spectral::Quadrature Mesh<Dim>::quadrature(
    const size_t d) const {
  return extract_quadrature(gsl::at(quadrature_and_basis_, d));
}

template <size_t Dim>
// clang-tidy: incorrectly reported redundancy in template expression
template <size_t N, Requires<(N > 0 and N == Dim)>>  // NOLINT
KOKKOS_FUNCTION Mesh<Dim - 1> Mesh<Dim>::slice_away(const size_t d) const {
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
// clang-tidy: incorrectly reported redundancy in template expression
template <size_t N, Requires<(N > 0 and N == Dim)>>  // NOLINT
KOKKOS_FUNCTION Mesh<Dim - 1> Mesh<Dim>::on_interface(const size_t d) const {
  auto face = slice_away(d);
  // The outer boundary of a B3 ball (B2 disk) is an S2 (S1): convert the
  // remaining ZernikeB3 (ZernikeB2) angular bases to SphericalHarmonic
  // (Fourier) so that the face mesh has the correct basis for the boundary.
  if (basis(d) == Spectral::Basis::ZernikeB3) {
    ASSERT(Dim == 3, "ZernikeB3 should only be used on 3D meshes, got " << Dim);
    ASSERT(quadrature(d) == Spectral::Quadrature::GaussRadauUpper,
           "A B3 ball's outer boundary is at the upper end of the radial "
           "dimension (GaussRadauUpper quadrature), but dimension "
               << d << " has quadrature " << quadrature(d));
    ASSERT(alg::all_of(basis(),
                       [](const Spectral::Basis b) {
                         return b == Spectral::Basis::ZernikeB3;
                       }),
           "Mesh is using ZernikeB3 basis in a nonisotropic manner, got "
               << basis());
    return {face.extents().indices(),
            make_array<Dim - 1>(Spectral::Basis::SphericalHarmonic),
            face.quadrature()};
  }
  if (basis(d) == Spectral::Basis::ZernikeB2) {
    ASSERT(Dim == 2 or Dim == 3,
           "ZernikeB2 should only be used on 2D or 3D meshes, got " << Dim);
    ASSERT(quadrature(d) == Spectral::Quadrature::GaussRadauUpper,
           "A B2 disk's outer boundary is at the upper end of the radial "
           "dimension (GaussRadauUpper quadrature), but dimension "
               << d << " has quadrature " << quadrature(d));
    auto bases = face.basis();
    ASSERT(alg::count(bases, Spectral::Basis::ZernikeB2) == 1,
           "Expected exactly one ZernikeB2 angular basis remaining in the "
           "face after slicing a B2 mesh, got face bases "
               << face.basis());
    for (auto& basis : bases) {
      if (basis == Spectral::Basis::ZernikeB2) {
        basis = Spectral::Basis::Fourier;
      }
    }
    return {face.extents().indices(), bases, face.quadrature()};
  }
  return face;
}

template <size_t Dim>
template <size_t SliceDim, Requires<(SliceDim <= Dim)>>
KOKKOS_FUNCTION Mesh<SliceDim> Mesh<Dim>::slice_through(
    const std::array<size_t, SliceDim>& dims) const {
  ASSERT(alg::none_of(
             dims, [&dims](const size_t d) { return alg::count(dims, d) > 1; }),
         "Dimensions to slice through contain duplicates.");
  Mesh<SliceDim> result{};
  for (size_t i = 0; i < SliceDim; ++i) {
    const auto& d = gsl::at(dims, i);
    ASSERT(d < Dim, "Tried to slice through non-existing dimension "
                        << d << " of " << Dim << "-dimensional mesh.");
    gsl::at(result.extents_, i) = gsl::at(extents_, d);
    gsl::at(result.quadrature_and_basis_, i) =
        gsl::at(quadrature_and_basis_, d);
  }
  return result;
}

template <size_t Dim>
KOKKOS_FUNCTION std::array<Mesh<1>, Dim> Mesh<Dim>::slices() const {
  std::array<Mesh<1>, Dim> result{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = Mesh<1>(extents(d), basis(d), quadrature(d));
  }
  return result;
}

template <size_t Dim>
void Mesh<Dim>::pup(PUP::er& p) {
  p | extents_;
  p | quadrature_and_basis_;
}

template <size_t Dim>
KOKKOS_FUNCTION bool is_isotropic(const Mesh<Dim>& mesh) {
  if constexpr (Dim == 0 or Dim == 1) {
    return true;
  } else {
    const auto all_equal_to_first = [](const auto& values) {
      return alg::all_of(values, [&values](const auto& value) {
        return value == gsl::at(values, 0);
      });
    };
    return all_equal_to_first(mesh.extents().indices()) and
           all_equal_to_first(mesh.basis()) and
           all_equal_to_first(mesh.quadrature());
  }
}

template <size_t Dim>
KOKKOS_FUNCTION bool operator==(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs) {
  return alg::equal(lhs.extents_, rhs.extents_) and
         alg::equal(lhs.quadrature_and_basis_, rhs.quadrature_and_basis_);
}

template <size_t Dim>
KOKKOS_FUNCTION bool operator!=(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs) {
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

#define INSTANTIATE_SLICE_AWAY(_, data)                                    \
  template Mesh<DIM(data) - 1> Mesh<DIM(data)>::slice_away(const size_t)   \
      const;                                                               \
  template Mesh<DIM(data) - 1> Mesh<DIM(data)>::on_interface(const size_t) \
      const;
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
