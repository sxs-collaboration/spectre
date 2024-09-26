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
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Poisson::Solutions {

namespace detail {
template <typename DataType, size_t Dim>
struct MoustacheVariables {
  using Cache = CachedTempBuffer<
      Tags::Field<DataType>,
      ::Tags::deriv<Tags::Field<DataType>, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Flux<Tags::Field<DataType>, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::FixedSource<Tags::Field<DataType>>>;

  const tnsr::I<DataType, Dim>& x;

  void operator()(gsl::not_null<Scalar<DataType>*> field,
                  gsl::not_null<Cache*> cache,
                  Tags::Field<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::i<DataType, Dim>*> field_gradient,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<Tags::Field<DataType>, tmpl::size_t<Dim>,
                                Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                  gsl::not_null<Cache*> cache,
                  ::Tags::Flux<Tags::Field<DataType>, tmpl::size_t<Dim>,
                               Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> fixed_source_for_field,
                  gsl::not_null<Cache*> cache,
                  ::Tags::FixedSource<Tags::Field<DataType>> /*meta*/) const;
};
}  // namespace detail

/*!
 * \brief A solution to the Poisson equation with a discontinuous first
 * derivative.
 *
 * \details This implements the solution \f$u(x,y)=x\left(1-x\right)
 * y\left(1-y\right)\left(\left(x-\frac{1}{2}\right)^2+\left(y-
 * \frac{1}{2}\right)^2\right)^\frac{3}{2}\f$ to the Poisson equation
 * in two dimensions, and
 * \f$u(x)=x\left(1-x\right)\left|x-\frac{1}{2}\right|^3\f$ in one dimension.
 * Their boundary conditions vanish on the square \f$[0,1]^2\f$ or interval
 * \f$[0,1]\f$, respectively.
 *
 * The corresponding source \f$f=-\Delta u\f$ has a discontinuous first
 * derivative at \f$\frac{1}{2}\f$. This accomplishes two things:
 *
 * - It makes it useful to test the convergence behaviour of our elliptic DG
 * solver.
 * - It makes it look like a moustache (at least in 1D).
 *
 * This solution is taken from \cite Stamm2010.
 */
template <size_t Dim>
class Moustache : public elliptic::analytic_data::AnalyticSolution {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "A solution with a discontinuous first derivative of its source at 1/2 "
      "that also happens to look like a moustache. It vanishes at zero and one "
      "in each dimension"};

  Moustache() = default;
  Moustache(const Moustache&) = default;
  Moustache& operator=(const Moustache&) = default;
  Moustache(Moustache&&) = default;
  Moustache& operator=(Moustache&&) = default;
  ~Moustache() override = default;
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<Moustache>(*this);
  }

  /// \cond
  explicit Moustache(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Moustache);  // NOLINT
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::MoustacheVariables<DataType, Dim>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{x};
    return {cache.get_var(computer, RequestedTags{})...};
  }
};

/// \cond
template <size_t Dim>
PUP::able::PUP_ID Moustache<Dim>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <size_t Dim>
constexpr bool operator==(const Moustache<Dim>& /*lhs*/,
                          const Moustache<Dim>& /*rhs*/) {
  return true;
}

template <size_t Dim>
constexpr bool operator!=(const Moustache<Dim>& lhs,
                          const Moustache<Dim>& rhs) {
  return not(lhs == rhs);
}

}  // namespace Poisson::Solutions
