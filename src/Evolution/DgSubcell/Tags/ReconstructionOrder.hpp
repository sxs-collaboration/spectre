// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

namespace evolution::dg::subcell::Tags {
/// The reconstruction order
///
/// Only set if the reconstruction method actually returns the reconstruction
/// order used. There can be a different reconstruction order in each logical
/// dimension of an element.
template <size_t Dim>
struct ReconstructionOrder : db::SimpleTag {
  using type = std::optional<tnsr::I<DataVector, Dim, Frame::ElementLogical>>;
};
}  // namespace evolution::dg::subcell::Tags
