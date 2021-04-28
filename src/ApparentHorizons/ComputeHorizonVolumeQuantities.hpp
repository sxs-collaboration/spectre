// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace ah {

/// Given the generalized harmonic variables in the volume, computes
/// the quantities that will be interpolated onto an apparent horizon.
///
/// This is meant to be the primary `compute_vars_to_interpolate`
/// for the horizon finder.
///
/// SrcTagList and DestTagList have limited flexibility, and their
/// restrictions are static_asserted inside the apply functions.  The
/// lack of complete flexibility is intentional, because most
/// computations (e.g. for observers) should be done only on the
/// horizon surface (i.e. after interpolation) as opposed to in the
/// volume; only those computations that require data in the volume
/// (e.g. volume numerical derivatives) should be done here.
///
/// For the dual-frame case, takes the Jacobians and does numerical
/// derivatives in order to avoid Hessians.
struct ComputeHorizonVolumeQuantities {
  // Single-frame case
  template <typename SrcTagList, typename DestTagList>
  static void apply(const gsl::not_null<Variables<DestTagList>*> target_vars,
                    const Variables<SrcTagList>& src_vars,
                    const Mesh<3>& mesh) noexcept;
  // Dual-frame case
  template <typename SrcTagList, typename DestTagList, typename TargetFrame>
  static void apply(
      const gsl::not_null<Variables<DestTagList>*> target_vars,
      const Variables<SrcTagList>& src_vars, const Mesh<3>& mesh,
      const Jacobian<DataVector, 3, TargetFrame, Frame::Inertial>& jacobian,
      const InverseJacobian<DataVector, 3, Frame::Logical, TargetFrame>&
          inverse_jacobian) noexcept;
};

}  // namespace ah
