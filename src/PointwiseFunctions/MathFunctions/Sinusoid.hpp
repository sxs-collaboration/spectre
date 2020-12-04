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
template <size_t VolumeDim, typename Fr>
class Sinusoid;

/*!
 *  \ingroup MathFunctionsGroup
 *  \brief Sinusoid \f$f = A \sin\left(k x + \delta \right)\f$.
 *
 *  \details Input file options are: Amplitude, Phase, and Wavenumber
 */
template <typename Fr>
class Sinusoid<1, Fr> : public MathFunction<1, Fr> {
 public:
  struct Amplitude {
    using type = double;
    static constexpr Options::String help = {"The amplitude."};
  };

  struct Wavenumber {
    using type = double;
    static constexpr Options::String help = {"The wavenumber."};
  };

  struct Phase {
    using type = double;
    static constexpr Options::String help = {"The phase shift."};
  };
  using options = tmpl::list<Amplitude, Wavenumber, Phase>;

  static constexpr Options::String help = {
      "Applies a Sinusoid function to the input value"};

  Sinusoid(double amplitude, double wavenumber, double phase) noexcept;

  Sinusoid() = default;
  ~Sinusoid() override = default;
  Sinusoid(const Sinusoid& /*rhs*/) = delete;
  Sinusoid& operator=(const Sinusoid& /*rhs*/) = delete;
  Sinusoid(Sinusoid&& /*rhs*/) noexcept = default;
  Sinusoid& operator=(Sinusoid&& /*rhs*/) noexcept = default;

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(MathFunction<1, Fr>),
                                     Sinusoid);  // NOLINT

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
  friend bool operator==(const Sinusoid& lhs, const Sinusoid& rhs) noexcept {
    return lhs.amplitude_ == rhs.amplitude_ and
           lhs.wavenumber_ == rhs.wavenumber_ and lhs.phase_ == rhs.phase_;
  }
  double amplitude_{};
  double wavenumber_{};
  double phase_{};

  template <typename T>
  T apply_call_operator(const T& x) const noexcept;
  template <typename T>
  T apply_first_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_second_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_third_deriv(const T& x) const noexcept;
};

template <typename Fr>
bool operator!=(const Sinusoid<1, Fr>& lhs,
                const Sinusoid<1, Fr>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace MathFunctions

/// \cond
template <typename Fr>
PUP::able::PUP_ID MathFunctions::Sinusoid<1, Fr>::my_PUP_ID = 0;  // NOLINT
/// \endcond
