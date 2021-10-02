// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/TagsDomain.hpp"

#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::domain::Tags {
template <size_t Dim>
void DivMeshVelocityCompute<Dim>::function(
    const gsl::not_null<std::optional<Scalar<DataVector>>*> div_mesh_velocity,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity,
    const ::Mesh<Dim>& mesh,
    const ::InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jac_logical_to_inertial) {
  if (mesh_velocity.has_value()) {
    *div_mesh_velocity =
        divergence(*mesh_velocity, mesh, inv_jac_logical_to_inertial);
    return;
  }
  *div_mesh_velocity = std::nullopt;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template struct DivMeshVelocityCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::domain::Tags
