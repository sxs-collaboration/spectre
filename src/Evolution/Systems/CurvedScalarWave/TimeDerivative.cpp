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
void TimeDerivative<Dim>::apply(
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
  *result_lapse = lapse;
  *result_shift = shift;
  *result_inverse_spatial_metric = upper_spatial_metric;
  *result_gamma1 = gamma1;
  *result_gamma2 = gamma2;

  tenex::evaluate(dt_psi,
                  -lapse() * pi() + shift(ti::I) * d_psi(ti::i) +
                      gamma1() * shift(ti::J) * (d_psi(ti::j) - phi(ti::j)));

  tenex::evaluate(
      dt_pi,
      lapse() * pi() * trace_extrinsic_curvature() +
          shift(ti::I) * d_pi(ti::i) +
          lapse() * trace_spatial_christoffel(ti::I) * phi(ti::i) +
          gamma1() * gamma2() * shift(ti::I) * (d_psi(ti::i) - phi(ti::i)) -
          lapse() * upper_spatial_metric(ti::I, ti::J) * d_phi(ti::i, ti::j) -
          upper_spatial_metric(ti::I, ti::J) * phi(ti::i) * deriv_lapse(ti::j));

  tenex::evaluate<ti::i>(
      dt_phi, -lapse() * d_pi(ti::i) + shift(ti::J) * d_phi(ti::j, ti::i) +
                  gamma2() * lapse() * (d_psi(ti::i) - phi(ti::i)) -
                  pi() * deriv_lapse(ti::i) +
                  phi(ti::j) * deriv_shift(ti::i, ti::J));
}
}  // namespace CurvedScalarWave
// Generate explicit instantiations of partial_derivatives function as well as
// all other functions in Equations.cpp

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep

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
