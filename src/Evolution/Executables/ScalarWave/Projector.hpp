// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Tags/NeighborMesh.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"
#include "Time/History.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

namespace amr {

template <size_t Dim, typename System>
struct Projector {
  using MortarKey = std::pair<Direction<Dim>, ElementId<Dim>>;
  template <typename MappedType>
  using MortarMap =
      std::unordered_map<MortarKey, MappedType, boost::hash<MortarKey>>;

  using variables_tag = typename System::variables_tag;
  using variables_t = typename variables_tag::type;

  using argument_tags = tmpl::list<::domain::Tags::Element<Dim>>;
  using return_tags =
      tmpl::list<::domain::Tags::Mesh<Dim>,
                 ::evolution::dg::Tags::NeighborMesh<Dim>, Tags::Flags<Dim>,
                 variables_tag, ::ScalarWave::Tags::ConstraintGamma2,
                 ::Tags::HistoryEvolvedVariables<variables_tag>,
                 ::evolution::dg::Tags::MortarMesh<Dim>,
                 ::evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>;

  static void apply(
      const gsl::not_null<Mesh<Dim>*> mesh,
      const gsl::not_null<
          typename ::evolution::dg::Tags::NeighborMesh<Dim>::type*>
          neighbor_meshes,
      const gsl::not_null<std::array<Flag, Dim>*> amr_flags,
      const gsl::not_null<variables_t*> vars,
      const gsl::not_null<Scalar<DataVector>*> gamma_2,
      const gsl::not_null<TimeSteppers::History<variables_t>*> history,
      const gsl::not_null<MortarMap<Mesh<Dim - 1>>*> mortar_meshes,
      const gsl::not_null<
          DirectionMap<Dim, std::optional<Variables<tmpl::list<
                                evolution::dg::Tags::MagnitudeOfNormal,
                                evolution::dg::Tags::NormalCovector<Dim>>>>>*>
          normal_covector_quantities,
      const Element<Dim>& element) {
    const Mesh<Dim> old_mesh = *mesh;
    *mesh = amr::projectors::mesh(old_mesh, *amr_flags);

    const auto projection_matrices =
        Spectral::p_projection_matrices(old_mesh, *mesh);

    auto projected_vars =
        apply_matrices(projection_matrices, *vars, old_mesh.extents());
    *vars = std::move(projected_vars);

    auto projected_gamma_2 =
        apply_matrices(projection_matrices, get(*gamma_2), old_mesh.extents());
    get(*gamma_2) = std::move(projected_gamma_2);

    history->map_entries([&projection_matrices, &old_mesh](const auto entry) {
      *entry = apply_matrices(projection_matrices, *entry, old_mesh.extents());
    });

    for (const auto& [direction, neighbors] : element.neighbors()) {
      (*normal_covector_quantities)[direction] = std::nullopt;
      for (const auto& neighbor : neighbors) {
        const auto mortar_id = std::make_pair(direction, neighbor);
        (*mortar_meshes)[mortar_id] = ::dg::mortar_mesh(
            mesh->slice_away(direction.dimension()),
            neighbor_meshes->at(mortar_id).slice_away(direction.dimension()));
      }
    }
    for (const auto& direction : element.external_boundaries()) {
      (*normal_covector_quantities)[direction] = std::nullopt;
    }
  }
};

}  // namespace amr
