// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace amr::Initialization {
/*!
 * \brief %Mutator meant to be used with
 *  `Initialization::Actions::AddSimpleTags`to initialize flags used for
 * adaptive mesh refinement
 *
 * DataBox:
 * - Adds:
 *   - `amr::domain::Tags::Flags<Dim>`
 *   - `amr::domain::Tags::NeighborFlags<Dim>`
 * - Removes: nothing
 * - Modifies: nothing
 *
 */
template <size_t Dim>
struct Initialize {
  using return_tags =
    tmpl::list<amr::domain::Tags::Flags<Dim>,
               amr::domain::Tags::NeighborFlags<Dim>>;
  using argument_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<std::array<amr::domain::Flag, Dim>*> amr_flags,
      const gsl::not_null<std::unordered_map<ElementId<Dim>,
          std::array<amr::domain::Flag, Dim>>*> /*neighbor_flags*/) {
    *amr_flags = make_array<Dim>(amr::domain::Flag::Undefined);
    // default initialization of NeighborFlags is okay
  }
};
}  // namespace amr::Initialization
