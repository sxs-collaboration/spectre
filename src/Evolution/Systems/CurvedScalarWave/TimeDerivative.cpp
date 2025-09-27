// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/TimeDerivative.hpp"

#include <array>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave {
template <size_t Dim>
evolution::dg::TimeDerivativeDecisions<Dim> TimeDerivative<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_phi,

    const gsl::not_null<Scalar<DataVector>*> result_lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> result_shift,
    const gsl::not_null<tnsr::II<DataVector, Dim>*>
        result_inverse_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> result_gamma1,
    const gsl::not_null<Scalar<DataVector>*> result_gamma2,

    const tnsr::i<DataVector, Dim>& d_psi, const tnsr::i<DataVector, Dim>& d_pi,
    const tnsr::ij<DataVector, Dim>& d_phi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim>& phi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim>& shift,
    const tnsr::i<DataVector, Dim>& deriv_lapse,
    const tnsr::iJ<DataVector, Dim>& deriv_shift,
    const tnsr::II<DataVector, Dim>& upper_spatial_metric,
    const tnsr::I<DataVector, Dim>& trace_spatial_christoffel,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2) {
  const auto& pi_times_lapse = result_lapse;
  tenex::evaluate(pi_times_lapse, pi() * lapse());
  const auto& d_psi_dot_shift = result_gamma1;
  get(*d_psi_dot_shift) = get<0>(d_psi) * get<0>(shift);
  for (size_t i = 1; i < Dim; ++i) {
    get(*d_psi_dot_shift) += d_psi.get(i) * shift.get(i);
  }
  get(*dt_psi) = -get(*result_lapse) + get(*d_psi_dot_shift);
  tenex::evaluate(
      dt_pi,
      (*result_lapse)() * trace_extrinsic_curvature() +
          shift(ti::I) * d_pi(ti::i) +
          lapse() * trace_spatial_christoffel(ti::I) * phi(ti::i) -
          lapse() * upper_spatial_metric(ti::I, ti::J) * d_phi(ti::i, ti::j) -
          upper_spatial_metric(ti::I, ti::J) * phi(ti::i) * deriv_lapse(ti::j));

  const auto& lapse_times_gamma2 = result_gamma2;
  get(*lapse_times_gamma2) = get(lapse) * get(gamma2);
  tenex::evaluate<ti::i>(
      dt_phi, -lapse() * d_pi(ti::i) + shift(ti::J) * d_phi(ti::j, ti::i) +
                  (*lapse_times_gamma2)() * (d_psi(ti::i) - phi(ti::i)) -
                  pi() * deriv_lapse(ti::i) +
                  phi(ti::j) * deriv_shift(ti::i, ti::J));

  // gamma1 is usually set to zero
  if (get(gamma1) != 0.) {
    for (size_t i = 0; i < Dim; ++i) {
      get(*d_psi_dot_shift) -= phi.get(i) * shift.get(i);
    }
    get(*d_psi_dot_shift) *= get(gamma1);
    get(*dt_psi) += get(*d_psi_dot_shift);
    get(*dt_pi) += get(gamma2) * get(*d_psi_dot_shift);
  }
  *result_lapse = lapse;
  *result_shift = shift;
  *result_inverse_spatial_metric = upper_spatial_metric;
  *result_gamma1 = gamma1;
  *result_gamma2 = gamma2;
  return {true};
}
}  // namespace CurvedScalarWave
// Generate explicit instantiations of partial_derivatives function as well as
// all other functions in Equations.cpp

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"

template <size_t Dim>
using derivative_tags = typename CurvedScalarWave::System<Dim>::gradients_tags;

template <size_t Dim>
using variables_tags =
    typename CurvedScalarWave::System<Dim>::variables_tag::tags_list;

using derivative_frame = Frame::Inertial;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                               \
  template class CurvedScalarWave::TimeDerivative<DIM(data)>;                \
  template Variables<                                                        \
      db::wrap_tags_in<::Tags::deriv, derivative_tags<DIM(data)>,            \
                       tmpl::size_t<DIM(data)>, derivative_frame>>           \
  partial_derivatives<derivative_tags<DIM(data)>, variables_tags<DIM(data)>, \
                      DIM(data), derivative_frame>(                          \
      const Variables<variables_tags<DIM(data)>>& u,                         \
      const Mesh<DIM(data)>& mesh,                                           \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,    \
                            derivative_frame>& inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
