// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/PrettyType.hpp"

namespace control_system::Actions {
/*!
 * \brief Simple action that will print the
 * `control_system::Tags::CurrentNumberOfMeasurements` for whatever control
 * system it is run on.
 */
struct PrintCurrentMeasurement {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) {
    const int current_measurement =
        db::get<control_system::Tags::CurrentNumberOfMeasurements>(box);

    Parallel::printf("%s: Current measurement = %d\n",
                     pretty_type::name<ParallelComponent>(),
                     current_measurement);
  }
};
}  // namespace control_system::Actions
