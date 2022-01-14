// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Events::Tags {
/// \brief The mesh for the observation computational grid. For hybrid methods
/// like DG-FD the observer mesh changes throughout the evolution.
template <size_t Dim>
struct ObserverMesh : db::SimpleTag {
  using type = ::Mesh<Dim>;
};

/// \brief Sets the `ObserverMesh` to `domain::Tags::Mesh`
///
/// This is what you would use for a single numerical method simulation. Hybrid
/// methods will supply their own tags.
template <size_t Dim>
struct ObserverMeshCompute : ObserverMesh<Dim>, db::ComputeTag {
  using base = ObserverMesh<Dim>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<::domain::Tags::Mesh<Dim>>;
  static void function(const gsl::not_null<return_type*> observer_mesh,
                       const ::Mesh<Dim>& mesh) {
    *observer_mesh = mesh;
  }
};
}  // namespace Events::Tags
