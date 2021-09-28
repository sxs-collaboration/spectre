// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace evolution::dg::subcell::Tags {
/// The mesh on the subcells
template <size_t VolumeDim>
struct Mesh : db::SimpleTag {
  static std::string name() { return "Subcell(Mesh)"; }
  using type = ::Mesh<VolumeDim>;
};
}  // namespace evolution::dg::subcell::Tags
