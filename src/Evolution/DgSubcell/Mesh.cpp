// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Mesh.hpp"

#include <array>
#include <cstddef>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell::fd {
template <size_t Dim>
Mesh<Dim> mesh(const Mesh<Dim>& dg_mesh) {
  ASSERT(dg_mesh.basis() == make_array<Dim>(Spectral::Basis::Legendre) or
             dg_mesh.basis() == make_array<Dim>(Spectral::Basis::Chebyshev),
         "The DG basis for computing the subcell mesh must be Legendre or "
         "Chebyshev but got DG mesh"
             << dg_mesh);
  ASSERT(dg_mesh.quadrature() == make_array<Dim>(Spectral::Quadrature::Gauss) or
             dg_mesh.quadrature() ==
                 make_array<Dim>(Spectral::Quadrature::GaussLobatto),
         "The DG quadrature for computing the subcell mesh must be Gauss or "
         "GaussLobatto but got DG mesh"
             << dg_mesh);
  std::array<size_t, Dim> extents{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(extents, d) = 2 * dg_mesh.extents(d) - 1;
  }
  return Mesh<Dim>{extents, Spectral::Basis::FiniteDifference,
                   Spectral::Quadrature::CellCentered};
}

template <size_t Dim>
Mesh<Dim> dg_mesh(const Mesh<Dim>& subcell_mesh, const Spectral::Basis basis,
                  const Spectral::Quadrature quadrature) {
  ASSERT(subcell_mesh.basis() ==
             make_array<Dim>(Spectral::Basis::FiniteDifference),
         "The basis for computing the DG mesh must be FiniteDifference but got "
             << subcell_mesh);
  ASSERT(
      subcell_mesh.quadrature() ==
          make_array<Dim>(Spectral::Quadrature::CellCentered),
      "The quadrature for computing the DG mesh must be CellCentered but got "
          << subcell_mesh);
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
  return Mesh<Dim>{extents, basis, quadrature};
}

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
