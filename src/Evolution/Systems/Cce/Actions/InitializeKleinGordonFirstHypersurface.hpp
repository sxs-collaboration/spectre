// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Given initial boundary data for the Klein-Gordon variable \f$\Psi\f$,
 * computes its initial hypersurface data.
 *
 * \details This action is to be called after boundary data has been received,
 * but before the time-stepping evolution loop. So, it should be either late in
 * an initialization phase or early (before a `Actions::Goto` loop or similar)
 * in the `Evolve` phase.
 */
struct InitializeKleinGordonFirstHypersurface {
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // In some contexts, this action may get re-run (e.g. self-start procedure)
    // In those cases, we do not want to alter the existing hypersurface data,
    // so we just exit. However, we do want to re-run the action each time
    // the self start 'reset's from the beginning
    if (db::get<::Tags::TimeStepId>(box).slab_number() > 0 or
        not db::get<::Tags::TimeStepId>(box).is_at_slab_boundary()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    db::mutate<Tags::KleinGordonPsi>(
        [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
               kg_psi_volume,
           const Scalar<SpinWeighted<ComplexDataVector, 0>>& boundary_kg_psi,
           const size_t l_max, const size_t number_of_radial_points) {
          const DataVector one_minus_y_collocation =
              1.0 -
              Spectral::collocation_points<Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto>(
                  number_of_radial_points);

          const size_t boundary_size =
              Spectral::Swsh::number_of_swsh_collocation_points(l_max);

          for (size_t i = 0; i < number_of_radial_points; i++) {
            ComplexDataVector angular_view_kg_psi{
                get(*kg_psi_volume).data().data() + boundary_size * i,
                boundary_size};

            // this gives psi = (boundary value) / r
            angular_view_kg_psi =
                get(boundary_kg_psi).data() * one_minus_y_collocation[i] / 2.;
          }
        },
        make_not_null(&box),
        db::get<Tags::BoundaryValue<Tags::KleinGordonPsi>>(box),
        db::get<Tags::LMax>(box), db::get<Tags::NumberOfRadialPoints>(box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Cce::Actions
