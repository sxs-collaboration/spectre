// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>

#include "Domain/Amr/Flag.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace amr {
/// \brief Information about an element that is communicated by AMR actions
///
/// \details amr::Actions::EvaluateRefinementCriteria and
/// amr::Actions::UpdateAmrDecision communicate the desired
/// amr::Flag%s and Mesh of an element.
template <size_t VolumeDim>
struct Info {
  std::array<Flag, VolumeDim> flags;
  Mesh<VolumeDim> new_mesh;

  /// Serialization for Charm++
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

/// Output operator for an Info.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Info<VolumeDim>& info);

template <size_t VolumeDim>
bool operator==(const Info<VolumeDim>& lhs, const Info<VolumeDim>& rhs);

template <size_t VolumeDim>
bool operator!=(const Info<VolumeDim>& lhs, const Info<VolumeDim>& rhs);
}  // namespace amr
