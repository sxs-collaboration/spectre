// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace evolution::dg::subcell::Tags {
/// \brief Computes the active mesh, which is the DG mesh if `ActiveGrid` is
/// `Dg` and the subcell mesh if `ActiveGrid` is `Subcell`.
template <size_t Dim>
struct ObserverMeshCompute : ::Events::Tags::ObserverMesh<Dim>, db::ComputeTag {
  using base = ::Events::Tags::ObserverMesh<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::domain::Tags::Mesh<Dim>, subcell::Tags::Mesh<Dim>,
                 subcell::Tags::ActiveGrid>;
  static void function(gsl::not_null<return_type*> active_mesh,
                       const ::Mesh<Dim>& dg_mesh,
                       const ::Mesh<Dim>& subcell_mesh,
                       const subcell::ActiveGrid active_grid);
};
}  // namespace evolution::dg::subcell::Tags
