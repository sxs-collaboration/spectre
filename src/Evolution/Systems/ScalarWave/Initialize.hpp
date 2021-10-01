// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Systems/ScalarWave/Constraints.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace ScalarWave {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Initialize items related to constraints of the ScalarWave system
///
/// We add both constraints and the constraint damping parameter to the
/// evolution databox.
///
/// DataBox changes:
/// - Adds:
///   * `ScalarWave::Tags::ConstraintGamma2`
/// - Removes: nothing
/// - Modifies: nothing
///
/// \note This action relies on the `SetupDataBox` aggregated initialization
/// mechanism, so `Actions::SetupDataBox` must be present in the
/// `Initialization` phase action list prior to this action.
template <size_t Dim>
struct InitializeConstraints {
  using simple_tags = tmpl::list<ScalarWave::Tags::ConstraintGamma2>;

  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    Scalar<DataVector> gamma_2{mesh.number_of_grid_points(), 0.};

    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::move(gamma_2));

    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace ScalarWave
