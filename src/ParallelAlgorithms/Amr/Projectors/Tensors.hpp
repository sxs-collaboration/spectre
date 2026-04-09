// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ChildSize.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::projectors {

/// \brief Update the Tensors corresponding to TensorTags after an AMR
/// change
///
/// There is a specialization for
/// `ProjectTensors<tmpl::list<TensorTags...>>` that can be used if a
/// `tmpl::list` is available.
///
/// \details For each item corresponding to each tag in TensorTags, project
/// the data for each tensor from the old mesh to the new mesh
///
/// \see ProjectVariables
template <size_t Dim, typename... TensorTags>
struct ProjectTensors : tt::ConformsTo<amr::protocols::Projector> {
  using return_tags = tmpl::list<TensorTags...>;
  using argument_tags =
      tmpl::list<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>>;

  static void apply(
      const gsl::not_null<typename TensorTags::type*>... tensors,
      const Element<Dim>& /*element*/, const Mesh<Dim>& new_mesh,
      const std::pair<Mesh<Dim>, Element<Dim>>& old_mesh_and_element) {
    const auto& old_mesh = old_mesh_and_element.first;
    if (old_mesh == new_mesh) {
      return;  // mesh was not refined, so no projection needed
    }
    const auto projection_matrices =
        Spectral::p_projection_matrices(old_mesh, new_mesh);
    const auto& old_extents = old_mesh.extents();
    const auto project_tensor = [&projection_matrices,
                                 old_extents](auto& tensor) {
      for (size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = apply_matrices(projection_matrices, tensor[i], old_extents);
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(project_tensor(*tensors));
  }

  template <typename... Tags>
  static void apply(const gsl::not_null<typename TensorTags::type*>... tensors,
                    const Element<Dim>& element, const Mesh<Dim>& new_mesh,
                    const tuples::TaggedTuple<Tags...>& parent_items) {
    const auto& element_id = element.id();
    const auto& parent_id = get<domain::Tags::Element<Dim>>(parent_items).id();
    const auto& parent_mesh = get<domain::Tags::Mesh<Dim>>(parent_items);
    const auto child_sizes =
        domain::child_size(element_id.segment_ids(), parent_id.segment_ids());
    const auto projection_matrices =
        Spectral::projection_matrix_parent_to_child(parent_mesh, new_mesh,
                                                    child_sizes);
    const auto project_tensor = [&](auto& new_tensor, const auto& old_tensor) {
      for (size_t i = 0; i < new_tensor.size(); ++i) {
        new_tensor[i] = apply_matrices(projection_matrices, old_tensor[i],
                                       parent_mesh.extents());
      }
      return 0;
    };
    expand_pack(project_tensor(*tensors, get<TensorTags>(parent_items))...);
  }

  template <typename... Tags>
  static void apply(
      const gsl::not_null<typename TensorTags::type*>... tensors,
      const Element<Dim>& element, const Mesh<Dim>& new_mesh,
      const std::unordered_map<ElementId<Dim>, tuples::TaggedTuple<Tags...>>&
          children_items) {
    const auto& element_id = element.id();
    bool first_child = true;
    for (const auto& [child_id, child_items] : children_items) {
      const auto& child_mesh = get<domain::Tags::Mesh<Dim>>(child_items);
      const auto child_sizes =
          domain::child_size(child_id.segment_ids(), element_id.segment_ids());
      const auto projection_matrices =
          Spectral::projection_matrix_child_to_parent(child_mesh, new_mesh,
                                                      child_sizes);
      if (first_child) {
        const auto project_tensor = [&](auto& new_tensor,
                                        const auto& old_tensor) {
          for (size_t i = 0; i < new_tensor.size(); ++i) {
            new_tensor[i] = apply_matrices(projection_matrices, old_tensor[i],
                                           child_mesh.extents());
          }
          return 0;
        };
        expand_pack(project_tensor(*tensors, get<TensorTags>(child_items))...);
        first_child = false;
      } else {
        const auto project_tensor = [&](auto& new_tensor,
                                        const auto& old_tensor) {
          for (size_t i = 0; i < new_tensor.size(); ++i) {
            new_tensor[i] += apply_matrices(projection_matrices, old_tensor[i],
                                            child_mesh.extents());
          }
          return 0;
        };
        expand_pack(project_tensor(*tensors, get<TensorTags>(child_items))...);
      }
    }
  }
};

/// \cond
template <size_t Dim, typename... TensorTags>
struct ProjectTensors<Dim, tmpl::list<TensorTags...>>
    : public ProjectTensors<Dim, TensorTags...> {};
/// \endcond
}  // namespace amr::projectors
