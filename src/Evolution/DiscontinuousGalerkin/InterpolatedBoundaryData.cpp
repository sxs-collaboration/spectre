// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/InterpolatedBoundaryData.hpp"

#include <cstddef>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <vector>

#include "Utilities/StdHelpers.hpp"

namespace evolution::dg {
template <size_t VolumeDim>
InterpolatedBoundaryData<VolumeDim>::InterpolatedBoundaryData(
    InterpolatedBoundaryData::Info info)
    : info_(std::move(info)) {}

template <size_t VolumeDim>
void InterpolatedBoundaryData<VolumeDim>::Info::pup(PUP::er& p) {
  p | data;
  p | target_mesh;
  p | offsets;
}

template <size_t VolumeDim>
void InterpolatedBoundaryData<VolumeDim>::pup(PUP::er& p) {
  p | info_;
}

template <size_t VolumeDim>
bool operator==(const InterpolatedBoundaryData<VolumeDim>& lhs,
                const InterpolatedBoundaryData<VolumeDim>& rhs) {
  return lhs.boundary_data() == rhs.boundary_data() and
         lhs.target_mesh() == rhs.target_mesh() and
         lhs.offsets() == rhs.offsets();
}

template <size_t VolumeDim>
bool operator!=(const InterpolatedBoundaryData<VolumeDim>& lhs,
                const InterpolatedBoundaryData<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const InterpolatedBoundaryData<VolumeDim>& value) {
  using ::operator<<;
  os << "boundary data = " << value.boundary_data()
     << "\ntarget mesh = " << value.target_mesh()
     << "\noffsets = " << value.offsets();
  return os;
}

template class InterpolatedBoundaryData<1>;
template class InterpolatedBoundaryData<2>;
template class InterpolatedBoundaryData<3>;
template bool operator==(const InterpolatedBoundaryData<1>&,
                         const InterpolatedBoundaryData<1>&);
template bool operator==(const InterpolatedBoundaryData<2>&,
                         const InterpolatedBoundaryData<2>&);
template bool operator==(const InterpolatedBoundaryData<3>&,
                         const InterpolatedBoundaryData<3>&);
template bool operator!=(const InterpolatedBoundaryData<1>&,
                         const InterpolatedBoundaryData<1>&);
template bool operator!=(const InterpolatedBoundaryData<2>&,
                         const InterpolatedBoundaryData<2>&);
template bool operator!=(const InterpolatedBoundaryData<3>&,
                         const InterpolatedBoundaryData<3>&);
template std::ostream& operator<<(std::ostream&,
                                  const InterpolatedBoundaryData<1>&);
template std::ostream& operator<<(std::ostream&,
                                  const InterpolatedBoundaryData<2>&);
template std::ostream& operator<<(std::ostream&,
                                  const InterpolatedBoundaryData<3>&);
}  // namespace evolution::dg
