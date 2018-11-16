// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines MathFunctions::Gaussian.

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
 *  \brief Gaussian \f$f = A \exp\left(-\frac{(x-x_0)^2}{w^2}\right)\f$.
 *
 *  \details Input file options are: Amplitude, Width, and Center
 */
class Gaussian : public MathFunction<1> {
 public:
  struct Amplitude {
    using type = double;
    static constexpr OptionString help = {"The amplitude."};
  };

  struct Width {
    using type = double;
    static constexpr OptionString help = {"The width."};
    static type lower_bound() noexcept { return 0.; }
  };

  struct Center {
    using type = double;
    static constexpr OptionString help = {"The center."};
  };
  using options = tmpl::list<Amplitude, Width, Center>;

  static constexpr OptionString help = {
      "Applies a Gaussian function to the input value"};

  WRAPPED_PUPable_decl_template(Gaussian);  // NOLINT

  explicit Gaussian(CkMigrateMessage* /*unused*/) noexcept {}

  Gaussian(double amplitude, double width, double center) noexcept;

  Gaussian() = default;
  ~Gaussian() override = default;
  Gaussian(const Gaussian& /*rhs*/) = delete;
  Gaussian& operator=(const Gaussian& /*rhs*/) = delete;
  Gaussian(Gaussian&& /*rhs*/) noexcept = default;
  Gaussian& operator=(Gaussian&& /*rhs*/) noexcept = default;

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
  friend bool operator==(const Gaussian& lhs, const Gaussian& rhs) noexcept;
  template <typename T>
  T apply_call_operator(const T& x) const noexcept;
  template <typename T>
  T apply_first_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_second_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_third_deriv(const T& x) const noexcept;

  double amplitude_{};
  double inverse_width_{};
  double center_{};
};

bool operator!=(const Gaussian& lhs, const Gaussian& rhs) noexcept;

}  // namespace MathFunctions
