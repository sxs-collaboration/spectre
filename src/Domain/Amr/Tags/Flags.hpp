// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"

/// \cond
namespace amr {
template <size_t VolumeDim>
struct Info;
}  // namespace amr
/// \endcond

namespace amr::Tags {
/// amr::Info for an Element.
template <size_t VolumeDim>
struct Info : db::SimpleTag {
  using type = amr::Info<VolumeDim>;
};

}  // namespace amr::Tags
