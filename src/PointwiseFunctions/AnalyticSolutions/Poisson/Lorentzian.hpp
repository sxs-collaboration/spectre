// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
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
struct LorentzianVariables {
  using Cache = CachedTempBuffer<
      Tags::Field<DataType>,
      ::Tags::deriv<Tags::Field<DataType>, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Flux<Tags::Field<DataType>, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::FixedSource<Tags::Field<DataType>>>;

  const tnsr::I<DataVector, Dim>& x;
  const double constant;
  const double complex_phase;

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
 * \brief A Lorentzian solution to the Poisson equation
 *
 * \details This implements the Lorentzian solution
 * \f$u(\boldsymbol{x})=\left(1+r^2\right)^{-\frac{1}{2}}\f$ to the
 * three-dimensional Poisson equation
 * \f$-\Delta u(\boldsymbol{x})=f(\boldsymbol{x})\f$, where
 * \f$r^2=x^2+y^2+z^2\f$. The corresponding source is
 * \f$f(\boldsymbol{x})=3\left(1+r^2\right)^{-\frac{5}{2}}\f$.
 *
 * If `DataType` is `ComplexDataVector`, the solution is multiplied by
 * `exp(i * complex_phase)` to rotate it in the complex plane. This allows to
 * use this solution for the complex Poisson equation.
 *
 * \note Corresponding 1D and 2D solutions are not implemented yet.
 */
template <size_t Dim, typename DataType = DataVector>
class Lorentzian : public elliptic::analytic_data::AnalyticSolution {
  static_assert(
      Dim == 3,
      "This solution is currently implemented in 3 spatial dimensions only");

 public:
  struct PlusConstant {
    using type = double;
    static constexpr Options::String help{"Constant added to the solution."};
  };

  struct ComplexPhase {
    using type = double;
    static constexpr Options::String help{
        "Phase 'phi' of a complex exponential 'exp(i phi)' that rotates the "
        "solution in the complex plane."};
  };

  using options = tmpl::flatten<tmpl::list<
      PlusConstant,
      tmpl::conditional_t<std::is_same_v<DataType, ComplexDataVector>,
                          ComplexPhase, tmpl::list<>>>>;
  static constexpr Options::String help{
      "A Lorentzian solution to the Poisson equation."};

  Lorentzian() = default;
  Lorentzian(const Lorentzian&) = default;
  Lorentzian& operator=(const Lorentzian&) = default;
  Lorentzian(Lorentzian&&) = default;
  Lorentzian& operator=(Lorentzian&&) = default;
  ~Lorentzian() override = default;

  explicit Lorentzian(const double constant, const double complex_phase = 0.)
      : constant_(constant), complex_phase_(complex_phase) {
    ASSERT((std::is_same_v<DataType, ComplexDataVector> or complex_phase == 0.),
           "The complex phase is only supported for ComplexDataVector.");
  }

  double constant() const { return constant_; }
  double complex_phase() const { return complex_phase_; }

  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<Lorentzian>(*this);
  }

  /// \cond
  explicit Lorentzian(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Lorentzian);  // NOLINT
  /// \endcond

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::LorentzianVariables<DataType, Dim>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{x, constant_, complex_phase_};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    p | constant_;
    p | complex_phase_;
  }

 private:
  double constant_ = std::numeric_limits<double>::signaling_NaN();
  double complex_phase_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <size_t Dim, typename DataType>
PUP::able::PUP_ID Lorentzian<Dim, DataType>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <size_t Dim, typename DataType>
bool operator==(const Lorentzian<Dim, DataType>& lhs,
                const Lorentzian<Dim, DataType>& rhs) {
  return lhs.constant() == rhs.constant() and
         lhs.complex_phase() == rhs.complex_phase();
}

template <size_t Dim, typename DataType>
bool operator!=(const Lorentzian<Dim, DataType>& lhs,
                const Lorentzian<Dim, DataType>& rhs) {
  return not(lhs == rhs);
}

}  // namespace Poisson::Solutions
