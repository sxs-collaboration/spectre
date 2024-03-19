// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace CurvedScalarWave::Worldtube::Initialization {

/*!
 * \brief Sets the initial value of `CurrentIteration to 0.
 */
struct InitializeCurrentIteration : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags =
      tmpl::list<CurvedScalarWave::Worldtube::Tags::CurrentIteration>;
  using argument_tags = tmpl::list<>;
  using simple_tags = return_tags;
  using compute_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  static void apply(const gsl::not_null<size_t*> current_iteration) {
    *current_iteration = 0;
  }
};
}  // namespace CurvedScalarWave::Worldtube::Initialization
