// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Mesh.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "Evolution/DiscontinuousGalerkin/NumberOfSubcellPoints.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell::fd {
template <size_t Dim>
bool dg_mesh_supports_subcell(const Mesh<Dim>& dg_mesh) {
  for (size_t d = 0; d < Dim; ++d) {
    const auto basis_d = dg_mesh.basis(d);
    if (basis_d != Spectral::Basis::Legendre and
        basis_d != Spectral::Basis::Chebyshev and
        basis_d != Spectral::Basis::Cartoon and
        basis_d != Spectral::Basis::ZernikeB1) {
      return false;
    }
  }
  return true;
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
Mesh<Dim> mesh(const Mesh<Dim>& dg_mesh,
               const std::array<int, 3>& number_of_points_coefficients) {
  if (not dg_mesh_supports_subcell(dg_mesh)) {
    // Non-hypercube topology.  Return uninitialized mesh: subcell should never
    // be used on this element.
    return Mesh<Dim>{0_st, Spectral::Basis::Uninitialized,
                     Spectral::Quadrature::Uninitialized};
  }
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
    if (dg_mesh.basis(d) == Spectral::Basis::Cartoon) {
      ASSERT(dg_mesh.extents(d) == 1,
             "A Cartoon dimension of the DG mesh must have exactly one grid "
             "point, but dimension "
                 << d << " of the DG mesh is " << dg_mesh);
      gsl::at(extents, d) = 1;
    } else {
      gsl::at(extents, d) = evolution::dg::number_of_subcell_points(
          dg_mesh.extents(d), number_of_points_coefficients);
    }
  }
  if constexpr (Dim == 3) {
    if (dg_mesh.basis(1) == Spectral::Basis::Cartoon and
        dg_mesh.basis(2) == Spectral::Basis::Cartoon) {
      ASSERT(dg_mesh.basis(0) == Spectral::Basis::Legendre or
                 dg_mesh.basis(0) == Spectral::Basis::Chebyshev or
                 dg_mesh.basis(0) == Spectral::Basis::ZernikeB1,
             "The DG mesh that is being converted to subcell can only mix "
             "Legendre, Chebyshev, or ZernikeB1 with Cartoon, but got "
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
      ASSERT((dg_mesh.slice_away(2).basis() ==
                  make_array<2>(Spectral::Basis::Legendre) or
              dg_mesh.slice_away(2).basis() ==
                  make_array<2>(Spectral::Basis::Chebyshev) or
              dg_mesh.slice_away(2).basis() ==
                  std::array{Spectral::Basis::ZernikeB1,
                             Spectral::Basis::Legendre} or
              dg_mesh.slice_away(2).basis() ==
                  std::array{Spectral::Basis::ZernikeB1,
                             Spectral::Basis::Chebyshev}),
             "The DG mesh that is being converted to subcell can only mix "
             "Legendre, Chebyshev, or ZernikeB1 with Cartoon, but got "
                 << dg_mesh);
      ASSERT(
          (dg_mesh.slice_away(2).quadrature() ==
               make_array<2>(Spectral::Quadrature::Gauss) or
           dg_mesh.slice_away(2).quadrature() ==
               make_array<2>(Spectral::Quadrature::GaussLobatto) or
           dg_mesh.slice_away(2).quadrature() ==
               std::array{Spectral::Quadrature::GaussRadauUpper,
                          Spectral::Quadrature::Gauss} or
           dg_mesh.slice_away(2).quadrature() ==
               std::array{Spectral::Quadrature::GaussRadauUpper,
                          Spectral::Quadrature::GaussLobatto}),
          "The DG quadrature for computing the subcell mesh must be Gauss or "
          "GaussLobatto (can have ZernikeB1 in first dimension) with a Cartoon "
          "quadrature but got DG mesh"
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

template bool dg_mesh_supports_subcell(const Mesh<1>& dg_mesh);
template bool dg_mesh_supports_subcell(const Mesh<2>& dg_mesh);
template bool dg_mesh_supports_subcell(const Mesh<3>& dg_mesh);
template void verify_subcell_mesh(const Mesh<1>& subcell_mesh,
                                  const bool neighbor);
template void verify_subcell_mesh(const Mesh<2>& subcell_mesh,
                                  const bool neighbor);
template void verify_subcell_mesh(const Mesh<3>& subcell_mesh,
                                  const bool neighbor);
template Mesh<1> mesh(const Mesh<1>& dg_mesh,
                      const std::array<int, 3>& number_of_points_coefficients);
template Mesh<2> mesh(const Mesh<2>& dg_mesh,
                      const std::array<int, 3>& number_of_points_coefficients);
template Mesh<3> mesh(const Mesh<3>& dg_mesh,
                      const std::array<int, 3>& number_of_points_coefficients);
}  // namespace evolution::dg::subcell::fd
