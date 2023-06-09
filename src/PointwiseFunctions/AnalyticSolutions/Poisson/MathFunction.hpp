// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Poisson::Solutions {

namespace detail {
template <typename DataType, size_t Dim>
struct MathFunctionVariables {
  using Cache = CachedTempBuffer<
      Tags::Field,
      ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Flux<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::FixedSource<Tags::Field>>;

  const tnsr::I<DataType, Dim>& x;
  const ::MathFunction<Dim, Frame::Inertial>& math_function;

  void operator()(gsl::not_null<Scalar<DataType>*> field,
                  gsl::not_null<Cache*> cache, Tags::Field /*meta*/) const;
  void operator()(gsl::not_null<tnsr::i<DataType, Dim>*> field_gradient,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>,
                                Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                  gsl::not_null<Cache*> cache,
                  ::Tags::Flux<Tags::Field, tmpl::size_t<Dim>,
                               Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> fixed_source_for_field,
                  gsl::not_null<Cache*> cache,
                  ::Tags::FixedSource<Tags::Field> /*meta*/) const;
};
}  // namespace detail

template <size_t Dim>
class MathFunction : public elliptic::analytic_data::AnalyticSolution {
 public:
  struct Function {
    using type = std::unique_ptr<::MathFunction<Dim, Frame::Inertial>>;
    static constexpr Options::String help = "The solution function";
  };

  using options = tmpl::list<Function>;
  static constexpr Options::String help{
      "Any solution to the Poisson equation given by a MathFunction "
      "implementation, such as a Gaussian."};

  MathFunction() = default;
  MathFunction(const MathFunction&) = delete;
  MathFunction& operator=(const MathFunction&) = delete;
  MathFunction(MathFunction&&) = default;
  MathFunction& operator=(MathFunction&&) = default;
  ~MathFunction() override = default;
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override;

  MathFunction(
      std::unique_ptr<::MathFunction<Dim, Frame::Inertial>> math_function);

  const ::MathFunction<Dim, Frame::Inertial>& math_function() const {
    return *math_function_;
  }

  /// \cond
  explicit MathFunction(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(MathFunction);  // NOLINT
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::MathFunctionVariables<DataType, Dim>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{x, *math_function_};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::unique_ptr<::MathFunction<Dim, Frame::Inertial>> math_function_;
};

template <size_t Dim>
bool operator==(const MathFunction<Dim>& lhs, const MathFunction<Dim>& rhs);

template <size_t Dim>
bool operator!=(const MathFunction<Dim>& lhs, const MathFunction<Dim>& rhs);

}  // namespace Poisson::Solutions
