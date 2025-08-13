// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarInterpolator.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace evolution::dg {
template <size_t VolumeDim>
MortarInfo<VolumeDim>::MortarInfo(MortarInfoData data)
    : data_(std::move(data)) {}

template <size_t VolumeDim>
void MortarInfo<VolumeDim>::MortarInfoData::pup(PUP::er& p) {
  p | interpolator;
  p | mortar_size;
  p | interface_data_policy;
  p | time_stepping_policy;
}

template <size_t VolumeDim>
void MortarInfo<VolumeDim>::pup(PUP::er& p) {
  p | data_;
}

template <size_t VolumeDim>
bool operator==(const MortarInfo<VolumeDim>& lhs,
                const MortarInfo<VolumeDim>& rhs) {
  return lhs.interpolator() == rhs.interpolator() and
         lhs.mortar_size() == rhs.mortar_size() and
         lhs.interface_data_policy() == rhs.interface_data_policy() and
         lhs.time_stepping_policy() == rhs.time_stepping_policy();
}

template <size_t VolumeDim>
bool operator!=(const MortarInfo<VolumeDim>& lhs,
                const MortarInfo<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const MortarInfo<VolumeDim>& mortar_info) {
  using ::operator<<;
  os << mortar_info.mortar_size() << ", " << mortar_info.interface_data_policy()
     << ", " << mortar_info.time_stepping_policy();
  return os;
}

template class MortarInfo<1>;
template class MortarInfo<2>;
template class MortarInfo<3>;
template bool operator==(const MortarInfo<1>&, const MortarInfo<1>&);
template bool operator==(const MortarInfo<2>&, const MortarInfo<2>&);
template bool operator==(const MortarInfo<3>&, const MortarInfo<3>&);
template bool operator!=(const MortarInfo<1>&, const MortarInfo<1>&);
template bool operator!=(const MortarInfo<2>&, const MortarInfo<2>&);
template bool operator!=(const MortarInfo<3>&, const MortarInfo<3>&);
template std::ostream& operator<<(std::ostream&, const MortarInfo<1>&);
template std::ostream& operator<<(std::ostream&, const MortarInfo<2>&);
template std::ostream& operator<<(std::ostream&, const MortarInfo<3>&);
}  // namespace evolution::dg
