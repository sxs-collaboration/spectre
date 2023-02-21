// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Poisson/MathFunction.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Poisson::Solutions {
namespace detail {

template <typename DataType, size_t Dim>
void MathFunctionVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> field,
    const gsl::not_null<Cache*> /*cache*/, Tags::Field /*meta*/) const {
  *field = math_function(x);
}

template <typename DataType, size_t Dim>
void MathFunctionVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*> field_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
    const {
  *field_gradient = math_function.first_deriv(x);
}

template <typename DataType, size_t Dim>
void MathFunctionVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
    const {
  const auto& field_gradient = cache->get_var(
      *this, ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>{});
  for (size_t d = 0; d < Dim; ++d) {
    flux_for_field->get(d) = field_gradient.get(d);
  }
}

template <typename DataType, size_t Dim>
void MathFunctionVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> fixed_source_for_field,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::Field> /*meta*/) const {
  const auto second_deriv = math_function.second_deriv(x);
  get(*fixed_source_for_field) = 0.;
  for (size_t d = 0; d < Dim; ++d) {
    get(*fixed_source_for_field) -= second_deriv.get(d, d);
  }
}

}  // namespace detail

template <size_t Dim>
PUP::able::PUP_ID MathFunction<Dim>::my_PUP_ID = 0;  // NOLINT

template <size_t Dim>
std::unique_ptr<elliptic::analytic_data::AnalyticSolution>
MathFunction<Dim>::get_clone() const {
  return std::make_unique<MathFunction>(math_function_->get_clone());
}

template <size_t Dim>
MathFunction<Dim>::MathFunction(
    std::unique_ptr<::MathFunction<Dim, Frame::Inertial>> math_function)
    : math_function_(std::move(math_function)) {}

template <size_t Dim>
MathFunction<Dim>::MathFunction(CkMigrateMessage* m)
    : elliptic::analytic_data::AnalyticSolution(m) {}

template <size_t Dim>
void MathFunction<Dim>::pup(PUP::er& p) {
  elliptic::analytic_data::AnalyticSolution::pup(p);
  p | math_function_;
}

template <size_t Dim>
bool operator==(const MathFunction<Dim>& lhs, const MathFunction<Dim>& rhs) {
  return lhs.math_function() == rhs.math_function();
}

template <size_t Dim>
bool operator!=(const MathFunction<Dim>& lhs, const MathFunction<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_VARS(_, data) \
  template class detail::MathFunctionVariables<DTYPE(data), DIM(data)>;
#define INSTANTIATE(_, data)                                    \
  template class MathFunction<DIM(data)>;                       \
  template bool operator==(const MathFunction<DIM(data)>& lhs,  \
                           const MathFunction<DIM(data)>& rhs); \
  template bool operator!=(const MathFunction<DIM(data)>& lhs,  \
                           const MathFunction<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATE_VARS, (1, 2, 3), (DataVector))
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef DTYPE
#undef INSTANTIATE_VARS
#undef INSTANTIATE

}  // namespace Poisson::Solutions
