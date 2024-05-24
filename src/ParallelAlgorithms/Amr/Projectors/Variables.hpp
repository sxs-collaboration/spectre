// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace amr::projectors {

/// \brief Update the Variables corresponding to VariablesTags after an AMR
/// change
///
/// There is a specialization for
/// `ProjectVariables<tmpl::list<VariablesTags...>>` that can be used if a
/// `tmpl::list` is available.
///
/// \details For each item corresponding to each tag in VariablesTags, project
/// the data for each variable from the old mesh to the new mesh
///
/// \see ProjectTensors
template <size_t Dim, typename... VariablesTags>
struct ProjectVariables : tt::ConformsTo<amr::protocols::Projector> {
  using return_tags = tmpl::list<VariablesTags...>;
  using argument_tags =
      tmpl::list<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>>;

  // p-refinement
  static void apply(
      const gsl::not_null<typename VariablesTags::type*>... vars,
      const Element<Dim>& /*element*/, const Mesh<Dim>& new_mesh,
      const std::pair<Mesh<Dim>, Element<Dim>>& old_mesh_and_element) {
    const auto& old_mesh = old_mesh_and_element.first;
    if (old_mesh == new_mesh) {
      return;  // mesh was not refined, so no projection needed
    }
    const auto projection_matrices =
        Spectral::p_projection_matrices(old_mesh, new_mesh);
    const auto& old_extents = old_mesh.extents();
    expand_pack(
        (*vars = apply_matrices(projection_matrices, *vars, old_extents))...);
  }

  // h-refinement
  template <typename... Tags>
  static void apply(const gsl::not_null<typename VariablesTags::type*>... vars,
                    const Element<Dim>& element, const Mesh<Dim>& child_mesh,
                    const tuples::TaggedTuple<Tags...>& parent_items) {
    const auto& element_id = element.id();
    const auto& parent_id = get<domain::Tags::Element<Dim>>(parent_items).id();
    const auto& parent_mesh = get<domain::Tags::Mesh<Dim>>(parent_items);
    std::array<Spectral::ChildSize, Dim> child_sizes{};
    for (size_t d = 0; d < Dim; ++d) {
      if (parent_id.segment_id(d) == element_id.segment_id(d)) {
        gsl::at(child_sizes, d) = Spectral::ChildSize::Full;
      } else if (parent_id.segment_id(d).id_of_child(Side::Lower) ==
                 element_id.segment_id(d)) {
        gsl::at(child_sizes, d) = Spectral::ChildSize::LowerHalf;
      } else if (parent_id.segment_id(d).id_of_child(Side::Upper) ==
                 element_id.segment_id(d)) {
        gsl::at(child_sizes, d) = Spectral::ChildSize::UpperHalf;
      } else {
        ERROR("Parent element " << parent_id << " is not a parent of element "
                                << element_id << ". Please report this bug.");
      }
    }
    const auto prolongation_matrices =
        Spectral::projection_matrix_parent_to_child(parent_mesh, child_mesh,
                                                    child_sizes);
    expand_pack((*vars = apply_matrices(prolongation_matrices,
                                        get<VariablesTags>(parent_items),
                                        parent_mesh.extents()))...);
  }

  // h-coarsening
  template <typename... Tags>
  static void apply(
      const gsl::not_null<typename VariablesTags::type*>... /*vars*/,
      const Element<Dim>& /*element*/, const Mesh<Dim>& /*new_mesh*/,
      const std::unordered_map<ElementId<Dim>, tuples::TaggedTuple<Tags...>>&
      /*children_items*/) {
    ERROR("h-coarsening not implemented yet");
  }
};

/// \cond
template <size_t Dim, typename... VariableTags>
struct ProjectVariables<Dim, tmpl::list<VariableTags...>>
    : public ProjectVariables<Dim, VariableTags...> {};
/// \endcond
}  // namespace amr::projectors
