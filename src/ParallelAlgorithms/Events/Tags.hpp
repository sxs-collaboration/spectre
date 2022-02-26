// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GetOutput.hpp"
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

/*!
 * \brief The coordinates used for observation.
 *
 * In methods like DG-FD the mesh and coordinates change throughout the
 * simulation, so we need to always grab the right ones.
 */
template <size_t Dim, typename Fr>
struct ObserverCoordinates : db::SimpleTag {
  static std::string name() { return get_output(Fr{}) + "Coordinates"; }
  using type = tnsr::I<DataVector, Dim, Fr>;
};

/// \brief Sets the `ObserverCoordinates` to `domain::Tags::Coordinates`
///
/// This is what you would use for a single numerical method simulation. Hybrid
/// methods will supply their own tags.
template <size_t Dim, typename Fr>
struct ObserverCoordinatesCompute : ObserverCoordinates<Dim, Fr>,
                                    db::ComputeTag {
  using base = ObserverCoordinates<Dim, Fr>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<::domain::Tags::Coordinates<Dim, Fr>>;
  static void function(const gsl::not_null<return_type*> observer_coords,
                       const return_type& coords) {
    for (size_t i = 0; i < Dim; ++i) {
      observer_coords->get(i).set_data_ref(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          make_not_null(&const_cast<DataVector&>(coords.get(i))));
    }
  }
};
}  // namespace Events::Tags
