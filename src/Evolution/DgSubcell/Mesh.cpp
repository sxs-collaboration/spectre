// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Mesh.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell::fd {
template <size_t Dim>
size_t get_computational_dim(const Mesh<Dim>& subcell_mesh) {
  if constexpr (Dim == 3) {
    if (subcell_mesh.quadrature(2) == Spectral::Quadrature::SphericalSymmetry) {
      return 1;
    } else if (subcell_mesh.quadrature(2) ==
               Spectral::Quadrature::AxialSymmetry) {
      return 2;
    } else {
      return 3;
    }
  } else {
    return Dim;
  }
}

template <size_t Dim>
size_t get_computational_dim(const Index<Dim>& subcell_extents) {
  if constexpr (Dim == 3) {
    if (subcell_extents[1] == 1) {
      return 1;
    } else if (subcell_extents[2] == 1) {
      return 2;
    } else {
      return 3;
    }
  } else {
    return Dim;
  }
}

template <size_t Dim>
void verify_subcell_mesh(const Mesh<Dim>& subcell_mesh, const bool neighbor) {
  const std::string neighbor_str = neighbor ? " neighbor" : "";
  if constexpr (Dim == 3) {
    if (subcell_mesh.quadrature(2) == Spectral::Quadrature::AxialSymmetry) {
      // Checking for spherical symmetry
      ASSERT(
          subcell_mesh.basis(2) == Spectral::Basis::Cartoon and
              subcell_mesh.basis(0) != Spectral::Basis::Cartoon,
          "Got a" << neighbor_str
                  << " basis with an invalid combination of Cartoon bases, got "
                  << subcell_mesh);
      ASSERT(subcell_mesh.slice_away(2) == Mesh<2>(subcell_mesh.extents(0),
                                                   subcell_mesh.basis(0),
                                                   subcell_mesh.quadrature(0)),
             "The non-cartoon"
                 << neighbor_str
                 << " subcell sub-mesh must have isotropic basis, "
                    "quadrature, and extents but got "
                 << subcell_mesh);
    } else if (subcell_mesh.quadrature(2) ==
               Spectral::Quadrature::SphericalSymmetry) {
      // Checking for axial symmetry
      ASSERT(
          subcell_mesh.slice_away(0) ==
              Mesh<2>(1, Spectral::Basis::Cartoon,
                      Spectral::Quadrature::SphericalSymmetry),
          "Got a" << neighbor_str
                  << " basis with an invalid combination of Cartoon bases, got "
                  << subcell_mesh);
    } else {
      ASSERT(
          Mesh<Dim>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                    subcell_mesh.quadrature(0)) == subcell_mesh,
          "The" << neighbor_str
                << " subcell mesh must have isotropic basis, quadrature, and "
                   "extents but got "
                << subcell_mesh);
    }
  } else {
    ASSERT(Mesh<Dim>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                     subcell_mesh.quadrature(0)) == subcell_mesh,
           "The" << neighbor_str
                 << " subcell mesh must have isotropic basis, quadrature, and "
                    "extents but got "
                 << subcell_mesh);
  }
}

template <size_t Dim>
void verify_subcell_extents(const Index<Dim>& subcell_extents,
                            const bool neighbor) {
  const std::string neighbor_str = neighbor ? " neighbor" : "";
  if constexpr (Dim == 3) {
    if (subcell_extents[1] == 1) {
      // Checking for spherical symmetry
      ASSERT(
          subcell_extents[0] != 1 and subcell_extents[2] == 1,
          "The" << neighbor_str
                << " subcell extents are neither isotropic nor a valid cartoon "
                   "pattern, got "
                << subcell_extents);
    } else if (subcell_extents[2] == 1) {
      // Checking for axial symmetry
      ASSERT(
          subcell_extents.slice_away(2) == Index<2>(subcell_extents[0]),
          "The" << neighbor_str
                << " subcell extents are neither isotropic nor a valid cartoon "
                   "pattern, got "
                << subcell_extents);
    } else {
      ASSERT(subcell_extents == Index<Dim>(subcell_extents[0]),
             "The" << neighbor_str << " subcell mesh must be uniform but is "
                   << subcell_extents);
    }
  } else {
    ASSERT(subcell_extents == Index<Dim>(subcell_extents[0]),
           "The" << neighbor_str << " subcell mesh must be uniform but is "
                 << subcell_extents);
  }
}

template <size_t Dim>
Mesh<Dim> mesh(const Mesh<Dim>& dg_mesh) {
  if (dg_mesh.basis(Dim - 1) != Spectral::Basis::Cartoon) {
    ASSERT(dg_mesh.basis() == make_array<Dim>(Spectral::Basis::Legendre) or
               dg_mesh.basis() == make_array<Dim>(Spectral::Basis::Chebyshev),
           "The DG basis for computing the subcell mesh must be Legendre or "
           "Chebyshev but got DG mesh"
               << dg_mesh);
    ASSERT(
        dg_mesh.quadrature() == make_array<Dim>(Spectral::Quadrature::Gauss) or
            dg_mesh.quadrature() ==
                make_array<Dim>(Spectral::Quadrature::GaussLobatto),
        "The DG quadrature for computing the subcell mesh must be Gauss or "
        "GaussLobatto but got DG mesh"
            << dg_mesh);
  }
  std::array<size_t, Dim> extents{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(extents, d) = 2 * dg_mesh.extents(d) - 1;
  }
  if constexpr (Dim == 3) {
    if (dg_mesh.basis(1) == Spectral::Basis::Cartoon and
        dg_mesh.basis(2) == Spectral::Basis::Cartoon) {
      ASSERT(dg_mesh.basis(0) == Spectral::Basis::Legendre or
                 dg_mesh.basis(0) == Spectral::Basis::Chebyshev,
             "The DG mesh that is being converted to subcell can only mix "
             "Legendre or Chebyshev with Cartoon, but got "
                 << dg_mesh);
      ASSERT(dg_mesh.slice_away(0).quadrature() ==
                 make_array<2>(Spectral::Quadrature::SphericalSymmetry),
             "Invalid combination of Cartoon bases, got " << dg_mesh);
      return Mesh<3>{extents,
                     {Spectral::Basis::FiniteDifference,
                      Spectral::Basis::Cartoon, Spectral::Basis::Cartoon},
                     {Spectral::Quadrature::CellCentered,
                      Spectral::Quadrature::SphericalSymmetry,
                      Spectral::Quadrature::SphericalSymmetry}};
    } else if (dg_mesh.basis(2) == Spectral::Basis::Cartoon) {
      ASSERT(dg_mesh.slice_away(2).basis() ==
                     make_array<2>(Spectral::Basis::Legendre) or
                 dg_mesh.slice_away(2).basis() ==
                     make_array<2>(Spectral::Basis::Chebyshev),
             "The DG mesh that is being converted to subcell can only mix "
             "Legendre or Chebyshev with Cartoon, but got "
                 << dg_mesh);
      ASSERT(
          dg_mesh.slice_away(2).quadrature() ==
                  make_array<2>(Spectral::Quadrature::Gauss) or
              dg_mesh.slice_away(2).quadrature() ==
                  make_array<2>(Spectral::Quadrature::GaussLobatto),
          "The DG quadrature for computing the subcell mesh must be Gauss or "
          "GaussLobatto with a Cartoon quadrature but got DG mesh"
              << dg_mesh);
      return Mesh<3>{
          extents,
          {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
           Spectral::Basis::Cartoon},
          {Spectral::Quadrature::CellCentered,
           Spectral::Quadrature::CellCentered,
           Spectral::Quadrature::AxialSymmetry}};
    }
  }
  return Mesh<Dim>{extents, Spectral::Basis::FiniteDifference,
                   Spectral::Quadrature::CellCentered};
}

template <size_t Dim>
Mesh<Dim> dg_mesh(const Mesh<Dim>& subcell_mesh, const Spectral::Basis basis,
                  const Spectral::Quadrature quadrature) {
  if (subcell_mesh.basis(Dim - 1) != Spectral::Basis::Cartoon) {
    ASSERT(
        subcell_mesh.basis() ==
            make_array<Dim>(Spectral::Basis::FiniteDifference),
        "The basis for computing the DG mesh must be FiniteDifference but got "
            << subcell_mesh);
    ASSERT(
        subcell_mesh.quadrature() ==
            make_array<Dim>(Spectral::Quadrature::CellCentered),
        "The quadrature for computing the DG mesh must be CellCentered but got "
            << subcell_mesh);
  }
  ASSERT(
      basis == Spectral::Basis::Legendre or basis == Spectral::Basis::Chebyshev,
      "The DG basis must be Legendre or Chebyshev but got " << basis);
  ASSERT(quadrature == Spectral::Quadrature::Gauss or
             quadrature == Spectral::Quadrature::GaussLobatto,
         "The DG quadrature for computing the DG mesh must be Gauss or "
         "GaussLobatto but "
             << quadrature);
  std::array<size_t, Dim> extents{};
  for (size_t d = 0; d < Dim; ++d) {
    ASSERT((subcell_mesh.extents(d) + 1) % 2 == 0,
           "Subcell mesh must have odd extents " << subcell_mesh);
    gsl::at(extents, d) = (subcell_mesh.extents(d) + 1) / 2;
  }
  if constexpr (Dim == 3) {
    if (subcell_mesh.basis(1) == Spectral::Basis::Cartoon and
        subcell_mesh.basis(2) == Spectral::Basis::Cartoon) {
      ASSERT(subcell_mesh.basis(0) == Spectral::Basis::FiniteDifference,
             "The basis for computing the DG mesh can only mix "
             "FiniteDifference with Cartoon, but got "
                 << subcell_mesh);
      ASSERT(subcell_mesh.slice_away(0).quadrature() ==
                 make_array<2>(Spectral::Quadrature::SphericalSymmetry),
             "Invalid combination of Cartoon bases, got " << subcell_mesh);
      return Mesh<3>{
          extents,
          {basis, Spectral::Basis::Cartoon, Spectral::Basis::Cartoon},
          {quadrature, Spectral::Quadrature::SphericalSymmetry,
           Spectral::Quadrature::SphericalSymmetry}};
    } else if (subcell_mesh.basis(2) == Spectral::Basis::Cartoon) {
      ASSERT(subcell_mesh.slice_away(2).basis() ==
                 make_array<2>(Spectral::Basis::FiniteDifference),
             "The basis for computing the DG mesh can only mix "
             "FiniteDifference with Cartoon, but got "
                 << subcell_mesh);
      ASSERT(subcell_mesh.slice_away(2).quadrature() ==
                 make_array<2>(Spectral::Quadrature::CellCentered),
             "The quadrature for computing the DG mesh, if not a Cartoon "
             "quadrature, must be CellCentered "
             "but got "
                 << subcell_mesh);
      return Mesh<3>{
          extents,
          {basis, basis, Spectral::Basis::Cartoon},
          {quadrature, quadrature, Spectral::Quadrature::AxialSymmetry}};
    }
  }
  return Mesh<Dim>{extents, basis, quadrature};
}

template size_t get_computational_dim(const Mesh<1>& subcell_mesh);
template size_t get_computational_dim(const Mesh<2>& subcell_mesh);
template size_t get_computational_dim(const Mesh<3>& subcell_mesh);
template size_t get_computational_dim(const Index<1>& subcell_extents);
template size_t get_computational_dim(const Index<2>& subcell_extents);
template size_t get_computational_dim(const Index<3>& subcell_extents);
template void verify_subcell_mesh(const Mesh<1>& subcell_mesh,
                                  const bool neighbor);
template void verify_subcell_mesh(const Mesh<2>& subcell_mesh,
                                  const bool neighbor);
template void verify_subcell_mesh(const Mesh<3>& subcell_mesh,
                                  const bool neighbor);
template void verify_subcell_extents(const Index<1>& subcell_extents,
                                     const bool neighbor);
template void verify_subcell_extents(const Index<2>& subcell_extents,
                                     const bool neighbor);
template void verify_subcell_extents(const Index<3>& subcell_extents,
                                     const bool neighbor);

template Mesh<1> mesh(const Mesh<1>& dg_mesh);
template Mesh<2> mesh(const Mesh<2>& dg_mesh);
template Mesh<3> mesh(const Mesh<3>& dg_mesh);
template Mesh<1> dg_mesh(const Mesh<1>& subcell_mesh,
                         const Spectral::Basis basis,
                         const Spectral::Quadrature quadrature);
template Mesh<2> dg_mesh(const Mesh<2>& subcell_mesh,
                         const Spectral::Basis basis,
                         const Spectral::Quadrature quadrature);
template Mesh<3> dg_mesh(const Mesh<3>& subcell_mesh,
                         const Spectral::Basis basis,
                         const Spectral::Quadrature quadrature);
}  // namespace evolution::dg::subcell::fd
