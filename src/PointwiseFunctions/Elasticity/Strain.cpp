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

template <typename DataType, size_t Dim>
void strain(const gsl::not_null<tnsr::ii<DataType, Dim>*> strain,
            const tnsr::iJ<DataType, Dim>& deriv_displacement) noexcept {
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      strain->get(i, j) =
          0.5 * (deriv_displacement.get(i, j) + deriv_displacement.get(j, i));
    }
  }
}

template <typename DataType, size_t Dim>
void strain(const gsl::not_null<tnsr::ii<DataType, Dim>*> strain,
            const tnsr::iJ<DataType, Dim>& deriv_displacement,
            const tnsr::ii<DataType, Dim>& metric,
            const tnsr::ijj<DataType, Dim>& deriv_metric,
            const tnsr::ijj<DataType, Dim>& christoffel_first_kind,
            const tnsr::I<DataType, Dim>& displacement) noexcept {
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      // Unroll k=0 iteration of the loop below to avoid filling the result with
      // zeros initially
      strain->get(i, j) =
          0.5 * (metric.get(j, 0) * deriv_displacement.get(i, 0) +
                 get<0>(displacement) * deriv_metric.get(i, j, 0) +
                 metric.get(i, 0) * deriv_displacement.get(j, 0) +
                 get<0>(displacement) * deriv_metric.get(j, i, 0)) -
          christoffel_first_kind.get(0, i, j) * get<0>(displacement);
      for (size_t k = 1; k < Dim; ++k) {
        strain->get(i, j) +=
            0.5 * (metric.get(j, k) * deriv_displacement.get(i, k) +
                   displacement.get(k) * deriv_metric.get(i, j, k) +
                   metric.get(i, k) * deriv_displacement.get(j, k) +
                   displacement.get(k) * deriv_metric.get(j, i, k)) -
            christoffel_first_kind.get(k, i, j) * displacement.get(k);
      }
    }
  }
}

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
  Elasticity::strain(strain, displacement_gradient);
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                       \
  template void strain<DIM(data)>(                                 \
      gsl::not_null<tnsr::ii<DataVector, DIM(data)>*> strain,      \
      const tnsr::I<DataVector, DIM(data)>& displacement,          \
      const Mesh<DIM(data)>& mesh,                                 \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical, \
                            Frame::Inertial>& inv_jacobian) noexcept;

#define INSTANTIATE_DTYPE(_, data)                                          \
  template void strain(                                                     \
      gsl::not_null<tnsr::ii<DTYPE(data), DIM(data)>*> strain,              \
      const tnsr::iJ<DTYPE(data), DIM(data)>& deriv_displacement) noexcept; \
  template void strain(                                                     \
      gsl::not_null<tnsr::ii<DTYPE(data), DIM(data)>*> strain,              \
      const tnsr::iJ<DTYPE(data), DIM(data)>& deriv_displacement,           \
      const tnsr::ii<DTYPE(data), DIM(data)>& metric,                       \
      const tnsr::ijj<DTYPE(data), DIM(data)>& deriv_metric,                \
      const tnsr::ijj<DTYPE(data), DIM(data)>& christoffel_first_kind,      \
      const tnsr::I<DTYPE(data), DIM(data)>& displacement) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE_DTYPE, (2, 3), (double, DataVector))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace Elasticity
