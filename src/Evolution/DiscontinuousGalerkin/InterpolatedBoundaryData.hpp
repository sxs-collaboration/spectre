// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace evolution::dg {
/// \brief Information sent by a non-conforming Element that interpolates its
/// boundary data to a subset of the points of the Element receiving this
///
/// \details The following information is sent:
/// - the interpolated data (as a DataVector representing a type-erased
///   Variables)
/// - the Mesh that was used to compute the target points of the boundary face
///   of the receiving Element.  This is sent so that the receiving Element can
///   check if the data was interpolated to the correct points.
/// - the offsets of the interpolated data with respect to the target Mesh
template <size_t VolumeDim>
class InterpolatedBoundaryData {
  struct Info {
    DataVector data{};
    Mesh<VolumeDim - 1> target_mesh{};
    std::vector<size_t> offsets{};
    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p);
  };

 public:
  InterpolatedBoundaryData() = default;
  InterpolatedBoundaryData(const InterpolatedBoundaryData&) = default;
  InterpolatedBoundaryData(InterpolatedBoundaryData&&) = default;
  InterpolatedBoundaryData& operator=(const InterpolatedBoundaryData&) =
      default;
  InterpolatedBoundaryData& operator=(InterpolatedBoundaryData&&) = default;
  ~InterpolatedBoundaryData() = default;

  explicit InterpolatedBoundaryData(InterpolatedBoundaryData::Info info);

  const DataVector& boundary_data() const { return info_.data; }
  const Mesh<VolumeDim - 1>& target_mesh() const { return info_.target_mesh; }
  const std::vector<size_t>& offsets() const { return info_.offsets; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  Info info_;
};

template <size_t VolumeDim>
bool operator==(const InterpolatedBoundaryData<VolumeDim>& lhs,
                const InterpolatedBoundaryData<VolumeDim>& rhs);

template <size_t VolumeDim>
bool operator!=(const InterpolatedBoundaryData<VolumeDim>& lhs,
                const InterpolatedBoundaryData<VolumeDim>& rhs);

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const InterpolatedBoundaryData<VolumeDim>& value);
}  // namespace evolution::dg
