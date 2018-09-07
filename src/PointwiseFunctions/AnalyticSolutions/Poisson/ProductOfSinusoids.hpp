// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "Elliptic/Systems/Poisson/Tags.hpp"    // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <typename T>
class DataVectorImpl;
using DataVector = DataVectorImpl<double>;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Poisson {
namespace Solutions {
/*!
 * \brief A product of sinusoids \f$u(\boldsymbol{x}) = \prod_i \sin(k_i x_i)\f$
 *
 * \details Solves the Poisson equation \f$-\Delta u(x)=f(x)\f$ for a source
 * \f$f(x)=\boldsymbol{k}^2\prod_i \sin(k_i x_i)\f$.
 */
template <size_t Dim>
class ProductOfSinusoids {
 public:
  struct WaveNumbers {
    using type = std::array<double, Dim>;
    static constexpr OptionString help{"The wave numbers of the sinusoids"};
  };

  using options = tmpl::list<WaveNumbers>;
  static constexpr OptionString help{
      "A product of sinusoids that are taken of a wave number times the "
      "coordinate in each dimension."};

  ProductOfSinusoids() = default;
  ProductOfSinusoids(const ProductOfSinusoids&) noexcept = delete;
  ProductOfSinusoids& operator=(const ProductOfSinusoids&) noexcept = delete;
  ProductOfSinusoids(ProductOfSinusoids&&) noexcept = default;
  ProductOfSinusoids& operator=(ProductOfSinusoids&&) noexcept = default;
  ~ProductOfSinusoids() noexcept = default;

  explicit ProductOfSinusoids(
      const std::array<double, Dim>& wave_numbers) noexcept;

  tuples::TaggedTuple<Field, AuxiliaryField<Dim>> field_variables(
      const tnsr::I<DataVector, Dim>& x) const noexcept;

  tuples::TaggedTuple<::Tags::Source<Field>,
                      ::Tags::Source<AuxiliaryField<Dim>>>
  source_variables(const tnsr::I<DataVector, Dim>& x) const noexcept;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

  const std::array<double, Dim>& wave_numbers() const noexcept {
    return wave_numbers_;
  }

 private:
  std::array<double, Dim> wave_numbers_{
      {std::numeric_limits<double>::signaling_NaN()}};
};

template <size_t Dim>
bool operator==(const ProductOfSinusoids<Dim>& lhs,
                          const ProductOfSinusoids<Dim>& rhs) noexcept {
  return lhs.wave_numbers() == rhs.wave_numbers();
}

template <size_t Dim>
bool operator!=(const ProductOfSinusoids<Dim>& lhs,
                          const ProductOfSinusoids<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Solutions
}  // namespace Poisson
