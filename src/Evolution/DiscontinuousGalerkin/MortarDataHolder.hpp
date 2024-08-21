// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iosfwd>

#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace evolution::dg {
/// \brief Data on each side of the mortar used to compute the boundary
/// correction for the DG scheme using global time stepping.
template <size_t Dim>
class MortarDataHolder {
 public:
  /// Access the data on the local side.
  /// @{
  const MortarData<Dim>& local() const { return local_data_; }
  MortarData<Dim>& local() { return local_data_; }
  /// @}

  /// Access the data on the neighbor side.
  /// @{
  const MortarData<Dim>& neighbor() const { return neighbor_data_; }
  MortarData<Dim>& neighbor() { return neighbor_data_; }
  /// @}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  MortarData<Dim> local_data_{};
  MortarData<Dim> neighbor_data_{};
};

template <size_t Dim>
bool operator==(const MortarDataHolder<Dim>& lhs,
                const MortarDataHolder<Dim>& rhs);

template <size_t Dim>
bool operator!=(const MortarDataHolder<Dim>& lhs,
                const MortarDataHolder<Dim>& rhs);

template <size_t Dim>
std::ostream& operator<<(std::ostream& os,
                         const MortarDataHolder<Dim>& mortar_data_holder);
}  // namespace evolution::dg
