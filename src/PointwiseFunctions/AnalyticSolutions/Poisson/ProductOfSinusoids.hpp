// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Poisson::Solutions {

namespace detail {
template <typename DataType, size_t Dim>
struct ProductOfSinusoidsVariables {
  using Cache = CachedTempBuffer<
      Tags::Field,
      ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Flux<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::FixedSource<Tags::Field>>;

  const tnsr::I<DataType, Dim>& x;
  const std::array<double, Dim>& wave_numbers;

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

/*!
 * \brief A product of sinusoids \f$u(\boldsymbol{x}) = \prod_i \sin(k_i x_i)\f$
 *
 * \details Solves the Poisson equation \f$-\Delta u(x)=f(x)\f$ for a source
 * \f$f(x)=\boldsymbol{k}^2\prod_i \sin(k_i x_i)\f$.
 */
template <size_t Dim>
class ProductOfSinusoids : public elliptic::analytic_data::AnalyticSolution {
 public:
  struct WaveNumbers {
    using type = std::array<double, Dim>;
    static constexpr Options::String help{"The wave numbers of the sinusoids"};
  };

  using options = tmpl::list<WaveNumbers>;
  static constexpr Options::String help{
      "A product of sinusoids that are taken of a wave number times the "
      "coordinate in each dimension."};

  ProductOfSinusoids() = default;
  ProductOfSinusoids(const ProductOfSinusoids&) = default;
  ProductOfSinusoids& operator=(const ProductOfSinusoids&) = default;
  ProductOfSinusoids(ProductOfSinusoids&&) = default;
  ProductOfSinusoids& operator=(ProductOfSinusoids&&) = default;
  ~ProductOfSinusoids() override = default;
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<ProductOfSinusoids>(*this);
  }

  /// \cond
  explicit ProductOfSinusoids(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ProductOfSinusoids);  // NOLINT
  /// \endcond

  explicit ProductOfSinusoids(const std::array<double, Dim>& wave_numbers)
      : wave_numbers_(wave_numbers) {}

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::ProductOfSinusoidsVariables<DataType, Dim>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{x, wave_numbers_};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    p | wave_numbers_;
  }

  const std::array<double, Dim>& wave_numbers() const { return wave_numbers_; }

 private:
  std::array<double, Dim> wave_numbers_{
      {std::numeric_limits<double>::signaling_NaN()}};
};

/// \cond
template <size_t Dim>
PUP::able::PUP_ID ProductOfSinusoids<Dim>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <size_t Dim>
bool operator==(const ProductOfSinusoids<Dim>& lhs,
                const ProductOfSinusoids<Dim>& rhs) {
  return lhs.wave_numbers() == rhs.wave_numbers();
}

template <size_t Dim>
bool operator!=(const ProductOfSinusoids<Dim>& lhs,
                const ProductOfSinusoids<Dim>& rhs) {
  return not(lhs == rhs);
}

}  // namespace Poisson::Solutions
