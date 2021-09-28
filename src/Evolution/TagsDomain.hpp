// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace evolution {
namespace domain {
namespace Tags {
/// The divergence of the frame velocity
template <size_t Dim>
struct DivMeshVelocityCompute : db::ComputeTag,
                                ::domain::Tags::DivMeshVelocity {
  using base = DivMeshVelocity;
  using return_type = typename base::type;

  static void function(
      gsl::not_null<std::optional<Scalar<DataVector>>*> div_mesh_velocity,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const ::Mesh<Dim>& mesh,
      const ::InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                              Frame::Inertial>& inv_jac_logical_to_inertial);

  using argument_tags =
      tmpl::list<::domain::Tags::MeshVelocity<Dim, Frame::Inertial>,
                 ::domain::Tags::Mesh<Dim>,
                 ::domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                 Frame::Inertial>>;
};
}  // namespace Tags
}  // namespace domain
}  // namespace evolution
