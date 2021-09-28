// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::Tags {

/// The `FieldsTag` on external boundaries
template <size_t Dim, typename FieldsTag>
struct BoundaryFieldsCompute : db::ComputeTag,
                               domain::Tags::Faces<Dim, FieldsTag> {
  using base = domain::Tags::Faces<Dim, FieldsTag>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<FieldsTag, domain::Tags::Mesh<Dim>,
                                   domain::Tags::Element<Dim>>;
  static void function(const gsl::not_null<return_type*> vars_on_face,
                       const typename FieldsTag::type& vars,
                       const Mesh<Dim>& mesh, const Element<Dim>& element) {
    ASSERT(mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto,
           "Slicing fields to the boundary currently supports only "
           "Gauss-Lobatto grids. Add support to "
           "'elliptic::Tags::BoundaryFieldsCompute'.");
    for (const auto& direction : element.external_boundaries()) {
      data_on_slice(make_not_null(&((*vars_on_face)[direction])), vars,
                    mesh.extents(), direction.dimension(),
                    index_to_slice_at(mesh.extents(), direction));
    }
  }
};

/// The `::Tags::NormalDotFlux<FieldsTag>` on external boundaries
template <size_t Dim, typename FieldsTag, typename FluxesTag>
struct BoundaryFluxesCompute
    : db::ComputeTag,
      domain::Tags::Faces<
          Dim, db::add_tag_prefix<::Tags::NormalDotFlux, FieldsTag>> {
  using base =
      domain::Tags::Faces<Dim,
                          db::add_tag_prefix<::Tags::NormalDotFlux, FieldsTag>>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<FluxesTag,
                 domain::Tags::Faces<Dim, domain::Tags::FaceNormal<Dim>>,
                 domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>>;
  static void function(
      const gsl::not_null<return_type*> normal_dot_fluxes,
      const typename FluxesTag::type& fluxes,
      const DirectionMap<Dim, tnsr::i<DataVector, Dim>>& face_normals,
      const Mesh<Dim>& mesh, const Element<Dim>& element) {
    ASSERT(mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto,
           "Slicing fluxes to the boundary currently supports only "
           "Gauss-Lobatto grids. Add support to "
           "'elliptic::Tags::BoundaryFluxesCompute'.");
    for (const auto& direction : element.external_boundaries()) {
      const auto fluxes_on_face =
          data_on_slice(fluxes, mesh.extents(), direction.dimension(),
                        index_to_slice_at(mesh.extents(), direction));
      normal_dot_flux(make_not_null(&((*normal_dot_fluxes)[direction])),
                      face_normals.at(direction), fluxes_on_face);
    }
  }
};

}  // namespace elliptic::Tags
