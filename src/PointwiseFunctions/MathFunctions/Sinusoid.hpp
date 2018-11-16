// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines MathFunctions::Sinusoid.

#pragma once

#include <pup.h>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace MathFunctions {

/*!
 *  \ingroup MathFunctionsGroup
 *  \brief Sinusoid \f$f = A \sin\left(k x + \delta \right)\f$.
 *
 *  \details Input file options are: Amplitude, Phase, and Wavenumber
 */
class Sinusoid : public MathFunction<1> {
 public:
  struct Amplitude {
    using type = double;
    static constexpr OptionString help = {"The amplitude."};
  };

  struct Wavenumber {
    using type = double;
    static constexpr OptionString help = {"The wavenumber."};
  };

  struct Phase {
    using type = double;
    static constexpr OptionString help = {"The phase shift."};
    static type default_value() noexcept { return 0.; }
  };
  using options = tmpl::list<Amplitude, Wavenumber, Phase>;

  static constexpr OptionString help = {
      "Applies a Sinusoid function to the input value"};

  Sinusoid(double amplitude, double wavenumber, double phase) noexcept;

  Sinusoid() = default;
  ~Sinusoid() override = default;
  Sinusoid(const Sinusoid& /*rhs*/) = delete;
  Sinusoid& operator=(const Sinusoid& /*rhs*/) = delete;
  Sinusoid(Sinusoid&& /*rhs*/) noexcept = default;
  Sinusoid& operator=(Sinusoid&& /*rhs*/) noexcept = default;

  WRAPPED_PUPable_decl_template(Sinusoid); //NOLINT

  explicit Sinusoid(CkMigrateMessage* /*unused*/) noexcept {}

  double operator()(const double& x) const noexcept override;
  DataVector operator()(const DataVector& x) const noexcept override;

  double first_deriv(const double& x) const noexcept override;
  DataVector first_deriv(const DataVector& x) const noexcept override;

  double second_deriv(const double& x) const noexcept override;
  DataVector second_deriv(const DataVector& x) const noexcept override;

  double third_deriv(const double& x) const noexcept override;
  DataVector third_deriv(const DataVector& x) const noexcept override;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override;  // NOLINT

 private:
  friend bool operator==(const Sinusoid& lhs, const Sinusoid& rhs) noexcept;
  template <typename T>
  T apply_call_operator(const T& x) const noexcept;
  template <typename T>
  T apply_first_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_second_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_third_deriv(const T& x) const noexcept;

  double amplitude_{};
  double wavenumber_{};
  double phase_{};
};

bool operator!=(const Sinusoid& lhs, const Sinusoid& rhs) noexcept;

}  // namespace MathFunctions
