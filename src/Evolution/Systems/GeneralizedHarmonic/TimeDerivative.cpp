// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.tpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "Utilities/GenerateInstantiations.hpp"

// Explicit instantiations of structs defined in `Equations.cpp` as well as of
// `partial_derivatives` function for use in the computation of spatial
// derivatives of `gradients_tags`, and of the initial gauge source function
// (needed in `Initialize.hpp`).
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"

using derivative_frame = Frame::Inertial;

template <size_t Dim>
using derivative_tags_initial_gauge =
    tmpl::list<gh::Tags::InitialGaugeH<DataVector, Dim, derivative_frame>>;

template <size_t Dim>
using variables_tags_initial_gauge =
    tmpl::list<gh::Tags::InitialGaugeH<DataVector, Dim, derivative_frame>>;

template <size_t Dim>
using derivative_tags = typename gh::System<Dim>::gradients_tags;

template <size_t Dim>
using variables_tags = typename gh::System<Dim>::variables_tag::tags_list;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                  \
  template struct gh::TimeDerivative<gh::Solutions::all_solutions<DIM(data)>, \
                                     DIM(data)>;                              \
  template Variables<                                                         \
      db::wrap_tags_in<::Tags::deriv, derivative_tags<DIM(data)>,             \
                       tmpl::size_t<DIM(data)>, derivative_frame>>            \
  partial_derivatives<derivative_tags<DIM(data)>, variables_tags<DIM(data)>,  \
                      DIM(data), derivative_frame>(                           \
      const Variables<variables_tags<DIM(data)>>& u,                          \
      const Mesh<DIM(data)>& mesh,                                            \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,     \
                            derivative_frame>& inverse_jacobian);             \
  template Variables<db::wrap_tags_in<                                        \
      ::Tags::deriv, derivative_tags_initial_gauge<DIM(data)>,                \
      tmpl::size_t<DIM(data)>, derivative_frame>>                             \
  partial_derivatives<derivative_tags_initial_gauge<DIM(data)>,               \
                      variables_tags_initial_gauge<DIM(data)>, DIM(data),     \
                      derivative_frame>(                                      \
      const Variables<variables_tags_initial_gauge<DIM(data)>>& u,            \
      const Mesh<DIM(data)>& mesh,                                            \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,     \
                            derivative_frame>& inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
