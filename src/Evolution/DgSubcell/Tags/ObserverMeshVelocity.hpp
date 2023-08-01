// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <size_t Dim>
class Mesh;
/// \endcond

namespace evolution::dg::subcell::Tags {
/*!
 * \brief Computes the mesh velocity on the active grid.
 */
template <size_t Dim>
struct ObserverMeshVelocityCompute
    : db::ComputeTag,
      ::Events::Tags::ObserverMeshVelocity<Dim, Frame::Inertial> {
  using base = ::Events::Tags::ObserverMeshVelocity<Dim, Frame::Inertial>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<ActiveGrid, ::domain::Tags::MeshVelocity<Dim, Frame::Inertial>,
                 ::domain::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::Mesh<Dim>>;

  static void function(
      gsl::not_null<return_type*> active_mesh_velocity,
      subcell::ActiveGrid active_grid,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          dg_mesh_velocity,
      const ::Mesh<Dim>& dg_mesh, const ::Mesh<Dim>& subcell_mesh);
};
}  // namespace evolution::dg::subcell::Tags
