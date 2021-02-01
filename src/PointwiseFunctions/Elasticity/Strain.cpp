// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Elasticity/Strain.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity {

template <size_t Dim>
void strain(const gsl::not_null<tnsr::ii<DataVector, Dim>*> strain,
            const tnsr::I<DataVector, Dim>& displacement, const Mesh<Dim>& mesh,
            const InverseJacobian<DataVector, Dim, Frame::Logical,
                                  Frame::Inertial>& inv_jacobian) noexcept {
  // Copy the displacement into a Variables to take partial derivatives because
  // at this time the `partial_derivatives` function only works with Variables.
  // This function is only used for observing the strain and derived quantities
  // (such as the potential energy) so performance isn't critical, but adding a
  // `partial_derivatives` overload that takes a Tensor is an obvious
  // optimization here.
  Variables<tmpl::list<Tags::Displacement<Dim>>> vars{
      mesh.number_of_grid_points()};
  get<Tags::Displacement<Dim>>(vars) = displacement;
  const auto displacement_gradient =
      get<::Tags::deriv<Tags::Displacement<Dim>, tmpl::size_t<Dim>,
                        Frame::Inertial>>(
          partial_derivatives<tmpl::list<Tags::Displacement<Dim>>>(
              vars, mesh, inv_jacobian));
  // Symmetrize
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      strain->get(i, j) = 0.5 * (displacement_gradient.get(i, j) +
                                 displacement_gradient.get(j, i));
    }
  }
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                       \
  template void strain<DIM(data)>(                                 \
      gsl::not_null<tnsr::ii<DataVector, DIM(data)>*> strain,      \
      const tnsr::I<DataVector, DIM(data)>& displacement,          \
      const Mesh<DIM(data)>& mesh,                                 \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical, \
                            Frame::Inertial>& inv_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace Elasticity
