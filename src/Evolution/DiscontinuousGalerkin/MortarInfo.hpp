// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>
#include <pup_stl.h>

#include "Evolution/DiscontinuousGalerkin/InterfaceDataPolicy.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdHelpers.hpp"

namespace evolution::dg {
/// \brief Information about the mortar between two Elements
///
/// \details The following information is stored:
/// - the InterfaceDataPolicy
/// - the mortar size; for conforming neighbors this is (in each dimension of
///   the mortar) the SegmentSize of the mortar with respect to the face of the
///   host Element
template <size_t VolumeDim>
class MortarInfo {
  struct MortarInfoData {
    std::array<Spectral::SegmentSize, VolumeDim - 1> mortar_size{};
    InterfaceDataPolicy policy{InterfaceDataPolicy::Uninitialized};
    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p);
  };

 public:
  MortarInfo() = default;
  MortarInfo(const MortarInfo&) = default;
  MortarInfo(MortarInfo&&) = default;
  MortarInfo& operator=(const MortarInfo&) = default;
  MortarInfo& operator=(MortarInfo&&) = default;
  ~MortarInfo() = default;

  explicit MortarInfo(MortarInfoData data);

  /// For conforming neighbors, the SegmentSize of the mortar with respect to
  /// the face of the host Element
  const std::array<Spectral::SegmentSize, VolumeDim - 1>& mortar_size() const {
    return data_.mortar_size;
  }

  /// The InterfaceDataPolicy of the host Element for the mortar
  const InterfaceDataPolicy& policy() const { return data_.policy; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  MortarInfoData data_;
};

template <size_t VolumeDim>
bool operator==(const MortarInfo<VolumeDim>& lhs,
                const MortarInfo<VolumeDim>& rhs);

template <size_t VolumeDim>
bool operator!=(const MortarInfo<VolumeDim>& lhs,
                const MortarInfo<VolumeDim>& rhs);

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const MortarInfo<VolumeDim>& mortar_info);
}  // namespace evolution::dg
