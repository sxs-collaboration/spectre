// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/BoundaryConditions/Cartoon.hpp"

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace domain::BoundaryConditions {
MarkAsCartoon::~MarkAsCartoon() = default;

bool is_cartoon(const std::unique_ptr<BoundaryCondition>& boundary_condition) {
  return dynamic_cast<const MarkAsCartoon* const>(boundary_condition.get()) !=
         nullptr;
}

template <size_t Dim>
bool dg_mesh_is_cartoon_compatible(const Mesh<Dim>& dg_mesh) {
  if constexpr (Dim == 3) {
    return (dg_mesh.basis(0) == Spectral::Basis::Legendre or
            dg_mesh.basis(0) == Spectral::Basis::Chebyshev or
            dg_mesh.basis(0) == Spectral::Basis::ZernikeB1) and
           (dg_mesh.basis(1) == Spectral::Basis::Legendre or
            dg_mesh.basis(1) == Spectral::Basis::Chebyshev or
            dg_mesh.basis(1) == Spectral::Basis::Cartoon) and
           dg_mesh.basis(2) == Spectral::Basis::Cartoon;
  } else {
    return false;
  }
}

template bool dg_mesh_is_cartoon_compatible(const Mesh<1>& dg_mesh);
template bool dg_mesh_is_cartoon_compatible(const Mesh<2>& dg_mesh);
template bool dg_mesh_is_cartoon_compatible(const Mesh<3>& dg_mesh);
}  // namespace domain::BoundaryConditions
