// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Mesh.hpp"

#include <array>
#include <cstddef>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell::fd {
template <size_t Dim>
Mesh<Dim> mesh(const Mesh<Dim>& dg_mesh) noexcept {
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

template Mesh<1> mesh(const Mesh<1>& dg_mesh) noexcept;
template Mesh<2> mesh(const Mesh<2>& dg_mesh) noexcept;
template Mesh<3> mesh(const Mesh<3>& dg_mesh) noexcept;
}  // namespace evolution::dg::subcell::fd
