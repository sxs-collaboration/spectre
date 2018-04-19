// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace dg {
namespace Actions {

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Compute \f$\hat{n} \cdot F\f$ on the boundaries for a
/// non-conservative system.
///
/// Uses:
/// - ConstGlobalCache: nothing
/// - DataBox:
///   Tags::InternalDirections<volume_dim>,
///   Tags::Interface<
///       Tags::InternalDirections<volume_dim>,
///       Tags::Normalized<Tags::UnnormalizedFaceNormal<volume_dim>,
///                        typename system::template magnitude_tag<
///                            Tags::UnnormalizedFaceNormal<volume_dim>>>>,
///   interface items as required by Metavariables::system::normal_dot_fluxes
///
/// DataBox changes:
/// - Adds:
///   Tags::Interface<
///       Tags::InternalDirections<volume_dim>,
///       db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>>
/// - Removes: nothing
/// - Modifies: nothing
struct ComputeNonconservativeBoundaryFluxes {
 private:
  template <typename Metavariables, typename... NormalDotFluxTags, size_t Dim,
            typename Frame, typename... Args>
  static void apply_flux(
      gsl::not_null<Variables<tmpl::list<NormalDotFluxTags...>>*> boundary_flux,
      const tnsr::i<DataVector, Dim, Frame>& unit_face_normal,
      const Args&... boundary_variables) noexcept {
    Metavariables::system::normal_dot_fluxes::apply(
        make_not_null(&get<NormalDotFluxTags>(*boundary_flux))...,
        boundary_variables..., unit_face_normal);
  }

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using variables_tag = typename system::variables_tag;

    using internal_directions_tag =
        Tags::InternalDirections<system::volume_dim>;

    using normal_tag = Tags::UnnormalizedFaceNormal<system::volume_dim>;

    using unit_normal_tag = Tags::Interface<
        internal_directions_tag,
        Tags::Normalized<normal_tag,
                         typename system::template magnitude_tag<normal_tag>>>;
    using interface_normal_dot_fluxes_tag =
        Tags::Interface<internal_directions_tag,
                        db::add_tag_prefix<Tags::NormalDotFlux, variables_tag>>;

    auto boundary_fluxes_result = db::apply<tmpl::push_front<
        tmpl::transform<
            typename Metavariables::system::normal_dot_fluxes::argument_tags,
            tmpl::bind<Tags::Interface, internal_directions_tag, tmpl::_1>>,
        internal_directions_tag, unit_normal_tag>>(
        [](const db::item_type<internal_directions_tag>& internal_directions,
           const db::item_type<unit_normal_tag>& unit_face_normals,
           const auto&... tensors) noexcept {
          db::item_type<interface_normal_dot_fluxes_tag> boundary_fluxes;

          for (const auto& direction : internal_directions) {
            const auto& side_unit_face_normal = unit_face_normals.at(direction);
            auto& side_boundary_flux = boundary_fluxes[direction];
            side_boundary_flux = std::decay_t<decltype(side_boundary_flux)>(
                side_unit_face_normal.begin()->size(), 0.0);
            apply_flux<Metavariables>(make_not_null(&side_boundary_flux),
                                      side_unit_face_normal,
                                      tensors.at(direction)...);
          }
          return boundary_fluxes;
        },
        box);

    return std::make_tuple(
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<interface_normal_dot_fluxes_tag>>(
            box, std::move(boundary_fluxes_result)));
  }
};
}  // namespace Actions
}  // namespace dg
