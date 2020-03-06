// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/optional.hpp>
#include <cstddef>
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
      gsl::not_null<boost::optional<Scalar<DataVector>>*> div_mesh_velocity,
      const boost::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const ::Mesh<Dim>& mesh,
      const ::InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
          inv_jac_logical_to_inertial) noexcept;

  using argument_tags = tmpl::list<
      ::domain::Tags::MeshVelocity<Dim, Frame::Inertial>,
      ::domain::Tags::Mesh<Dim>,
      ::domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>>;
};
}  // namespace Tags
}  // namespace domain
}  // namespace evolution
