// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic::Actions {

/*!
 * \brief Initialize the "fixed sources" of the elliptic equations, i.e. their
 * variable-independent source term \f$f(x)\f$
 *
 * This action initializes \f$f(x)\f$ in an elliptic system of PDEs \f$-div(F) +
 * S = f(x)\f$.
 *
 * Uses:
 * - System:
 *   - `primal_fields`
 * - DataBox:
 *   - `BackgroundTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `db::wrap_tags_in<::Tags::FixedSource, primal_fields>`
 */
template <typename System, typename BackgroundTag,
          size_t Dim = System::volume_dim>
struct InitializeFixedSources : tt::ConformsTo<::amr::protocols::Projector> {
 private:
  using fixed_sources_tag = ::Tags::Variables<
      db::wrap_tags_in<::Tags::FixedSource, typename System::primal_fields>>;

 public:  // Iterable action
  using const_global_cache_tags =
      tmpl::list<elliptic::dg::Tags::Massive, BackgroundTag>;
  using simple_tags = tmpl::list<fixed_sources_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<InitializeFixedSources>(make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

 public:  // DataBox mutator, amr::protocols::Projector
  using return_tags = tmpl::list<fixed_sources_tag>;
  using argument_tags = tmpl::list<
      domain::Tags::Coordinates<Dim, Frame::Inertial>, BackgroundTag,
      elliptic::dg::Tags::Massive, domain::Tags::Mesh<Dim>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      Parallel::Tags::Metavariables>;

  template <typename Background, typename Metavariables, typename... AmrData>
  static void apply(
      const gsl::not_null<typename fixed_sources_tag::type*> fixed_sources,
      const tnsr::I<DataVector, Dim>& inertial_coords,
      const Background& background, const bool massive, const Mesh<Dim>& mesh,
      const Scalar<DataVector>& det_inv_jacobian, const Metavariables& /*meta*/,
      const AmrData&... /*amr_data*/) {
    // Retrieve the fixed-sources of the elliptic system from the background,
    // which (along with the boundary conditions) define the problem we want to
    // solve.
    using factory_classes =
        typename std::decay_t<Metavariables>::factory_creation::factory_classes;
    *fixed_sources =
        call_with_dynamic_type<Variables<typename fixed_sources_tag::tags_list>,
                               tmpl::at<factory_classes, Background>>(
            &background, [&inertial_coords](const auto* const derived) {
              return variables_from_tagged_tuple(derived->variables(
                  inertial_coords, typename fixed_sources_tag::tags_list{}));
            });

    // Apply DG mass matrix to the fixed sources if the DG operator is massive
    if (massive) {
      *fixed_sources /= get(det_inv_jacobian);
      ::dg::apply_mass_matrix(fixed_sources, mesh);
    }
  }
};

}  // namespace elliptic::Actions
