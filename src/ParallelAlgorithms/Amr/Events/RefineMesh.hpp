// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <typeinfo>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Policies/EnforcePolicies.hpp"
#include "ParallelAlgorithms/Amr/Policies/Tags.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace amr::Events {
namespace detail {
template <typename Criterion>
struct get_tags {
  using type = typename Criterion::compute_tags_for_observation_box;
};

}  // namespace detail
/// \ingroup AmrGroup
/// \brief Performs p-refinement on the domain
///
/// \details
/// - Loops over all refinement criteria specified in the
///   input file, ignoring any requests to join or split the Element.
///   If no valid p-refinement decision is requested, no change is
///   made to the Element.
/// - Updates the Mesh and all return tags of Metavariables::amr::projectors
///
/// \warning This does not communicate the new Mesh to its neighbors, nor does
/// it update ::domain::Tags::NeighborMesh
class RefineMesh : public Event {
 public:
  /// \cond
  explicit RefineMesh(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(RefineMesh);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help = {"Perform p-refinement"};

  RefineMesh();

  using compute_tags_for_observation_box = tmpl::list<>;
  using return_tags = tmpl::list<::Tags::DataBox>;
  using argument_tags = tmpl::list<>;

  template <typename DbTags, typename Metavariables, typename Component>
  void operator()(const gsl::not_null<db::DataBox<DbTags>*> box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<Metavariables::volume_dim>& element_id,
                  const Component* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    // Evaluate AMR refinement criteria
    // NOTE: This evaluates all criteria.  In the future this could be
    // restricted to evaluate only those criteria that do p-refinement that can
    // be evaluated locally (i.e. do not need neighbor information that is
    // already in the DataBox)
    constexpr size_t volume_dim = Metavariables::volume_dim;
    auto overall_decision = make_array<volume_dim>(amr::Flag::Undefined);

    using compute_tags = tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 Criterion>,
        detail::get_tags<tmpl::_1>>>>;
    auto observation_box = make_observation_box<compute_tags>(box);

    const auto& refinement_criteria =
        db::get<amr::Criteria::Tags::Criteria>(*box);
    for (const auto& criterion : refinement_criteria) {
      auto decision = criterion->evaluate(observation_box, cache, element_id);
      for (size_t d = 0; d < volume_dim; ++d) {
        // Ignore h-refinement decisions
        if (decision[d] == amr::Flag::Split or decision[d] == amr::Flag::Join) {
          ERROR("The criterion '" << typeid(*criterion).name()
                                  << "' requested h-refinement, but RefineMesh "
                                     "only works for p-refinement.");
        } else {
          overall_decision[d] = std::max(overall_decision[d], decision[d]);
        }
      }
    }
    // If no refinement criteria requested p-refinement, then set flag to
    // do nothing
    for (size_t d = 0; d < volume_dim; ++d) {
      if (overall_decision[d] == amr::Flag::Undefined) {
        overall_decision[d] = amr::Flag::DoNothing;
      }
    }

    // Now p-refine
    const auto old_mesh_and_element =
        std::make_pair(db::get<::domain::Tags::Mesh<volume_dim>>(*box),
                       db::get<::domain::Tags::Element<volume_dim>>(*box));
    const auto& old_mesh = old_mesh_and_element.first;
    const auto& verbosity =
        db::get<logging::Tags::Verbosity<amr::OptionTags::AmrGroup>>(*box);

    // Enforce policies
    const auto& policies = db::get<amr::Tags::Policies>(*box);
    amr::enforce_policies(make_not_null(&overall_decision), policies,
                          element_id, old_mesh);

    if (alg::any_of(overall_decision, [](amr::Flag flag) {
          return (flag == amr::Flag::IncreaseResolution or
                  flag == amr::Flag::DecreaseResolution);
        })) {
      db::mutate<::domain::Tags::Mesh<volume_dim>>(
          [&old_mesh,
           &overall_decision](const gsl::not_null<Mesh<volume_dim>*> mesh) {
            *mesh = amr::projectors::mesh(old_mesh, overall_decision);
          },
          box);

      if (verbosity >= Verbosity::Debug) {
        Parallel::printf(
            "Increasing order of element %s: %s -> %s\n", element_id,
            old_mesh.extents(),
            db::get<::domain::Tags::Mesh<volume_dim>>(*box).extents());
      }

      // Run the projectors.
      tmpl::for_each<typename Metavariables::amr::projectors>(
          [&box, &old_mesh_and_element](auto projector_v) {
            using projector = typename decltype(projector_v)::type;
            try {
              db::mutate_apply<projector>(box, old_mesh_and_element);
            } catch (std::exception& e) {
              ERROR("Error in AMR projector '"
                    << pretty_type::get_name<projector>() << "':\n"
                    << e.what());
            }
          });
    }
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;
};
}  // namespace amr::Events
