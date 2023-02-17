// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Initialization {
/// \ingroup InitializationGroup
/// \brief Initialize items related to adaptive mesh refinement
///
/// \see InitializeItems
template <size_t Dim>
struct Initialize {
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;

  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<amr::Tags::Flags<Dim>>;
  using simple_tags =
      tmpl::push_back<return_tags, amr::Tags::NeighborFlags<Dim>>;

  using compute_tags = tmpl::list<>;

  /// Given the items fetched from a DataBox by the argument_tags, mutate
  /// the items in the DataBox corresponding to return_tags
  static void apply(
      const gsl::not_null<std::array<amr::Flag, Dim>*> amr_flags) {
    *amr_flags = make_array<Dim>(amr::Flag::Undefined);
  }
};
}  // namespace amr::Initialization
