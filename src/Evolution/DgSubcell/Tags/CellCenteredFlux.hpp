// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::Tags {
/// \brief Holds the cell-centered fluxes on the subcell mesh.
///
/// These are only needed when using high-order FD methods and when we are
/// actively doing FD. If either of these conditions isn't met, the value is
/// `std::nullopt`.
///
/// The cell-centered fluxes are stored in the DataBox so they can be computed
/// and sent to neighbor elements without having to be recomputed after
/// receiving neighbor data. This means we maintain a single-send algorithm
/// despite going to high-order, with no additional FLOP overhead. We do,
/// however, have the memory overhead of the cell-centered fluxes.
///
/// \note The `TagsList` is a list of the variables, not the fluxes. They are
/// wrapped in the `::Tags::Flux` prefix.
template <typename TagsList, size_t Dim, typename Fr = Frame::Inertial>
struct CellCenteredFlux : db::SimpleTag {
  using type = std::optional<Variables<
      db::wrap_tags_in<::Tags::Flux, TagsList, tmpl::size_t<Dim>, Fr>>>;
};
}  // namespace evolution::dg::subcell::Tags
