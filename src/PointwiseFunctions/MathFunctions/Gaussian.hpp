// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines MathFunctions::Gaussian.

#pragma once

#include <array>
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
class Gaussian;

/*!
 *  \ingroup MathFunctionsGroup
 *  \brief 1D Gaussian \f$f = A \exp\left(-\frac{(x-x_0)^2}{w^2}\right)\f$
 *
 *  \details Input file options are: Amplitude, Width, and Center. The function
 *  takes input of type `double` or `DataVector` and returns
 *  the same type as the input type.
 */
template <typename Fr>
class Gaussian<1, Fr> : public MathFunction<1, Fr> {
 public:
  struct Amplitude {
    using type = double;
    static constexpr Options::String help = {"The amplitude."};
  };

  struct Width {
    using type = double;
    static constexpr Options::String help = {"The width."};
    static type lower_bound() noexcept { return 0.; }
  };

  struct Center {
    using type = double;
    static constexpr Options::String help = {"The center."};
  };
  using options = tmpl::list<Amplitude, Width, Center>;

  static constexpr Options::String help = {
      "Computes a Gaussian about an arbitrary coordinate center with given "
      "width and amplitude"};

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(MathFunction<1, Fr>),
                                     Gaussian);  // NOLINT

  explicit Gaussian(CkMigrateMessage* /*unused*/) noexcept {}

  Gaussian(double amplitude, double width, double center) noexcept;
  Gaussian(double amplitude, double width,
           const std::array<double, 1>& center) noexcept;

  Gaussian() = default;
  ~Gaussian() override = default;
  Gaussian(const Gaussian& /*rhs*/) = delete;
  Gaussian& operator=(const Gaussian& /*rhs*/) = delete;
  Gaussian(Gaussian&& /*rhs*/) noexcept = default;
  Gaussian& operator=(Gaussian&& /*rhs*/) noexcept = default;

  double operator()(const double& x) const noexcept override;
  DataVector operator()(const DataVector& x) const noexcept override;
  using MathFunction<1, Fr>::operator();

  double first_deriv(const double& x) const noexcept override;
  DataVector first_deriv(const DataVector& x) const noexcept override;
  using MathFunction<1, Fr>::first_deriv;

  double second_deriv(const double& x) const noexcept override;
  DataVector second_deriv(const DataVector& x) const noexcept override;
  using MathFunction<1, Fr>::second_deriv;

  double third_deriv(const double& x) const noexcept override;
  DataVector third_deriv(const DataVector& x) const noexcept override;
  using MathFunction<1, Fr>::third_deriv;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override;  // NOLINT

 private:
  double amplitude_{};
  double inverse_width_{};
  double center_{};
  friend bool operator==(const Gaussian<1, Fr>& lhs,
                         const Gaussian<1, Fr>& rhs) noexcept {
    return lhs.amplitude_ == rhs.amplitude_ and
           lhs.inverse_width_ == rhs.inverse_width_ and
           lhs.center_ == rhs.center_;
  }

  template <typename T>
  T apply_call_operator(const T& x) const noexcept;
  template <typename T>
  T apply_first_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_second_deriv(const T& x) const noexcept;
  template <typename T>
  T apply_third_deriv(const T& x) const noexcept;
};

/*!
 * \ingroup MathFunctionsGroup
 * \brief Gaussian \f$f = A \exp\left(-\frac{(x-x_0)^2}{w^2}\right)\f$
 *
 * \details Input file options are: Amplitude, Width, and Center. The function
 * takes input coordinates of type `tnsr::I<T, VolumeDim, Fr>`, where `T` is
 * e.g. `double` or `DataVector`, `Fr` is a frame (e.g. `Frame::Inertial`), and
 * `VolumeDim` is the dimension of the spatial volume, i.e. 2 or 3. (The case of
 * VolumeDim == 1 is handled specially by Gaussian<1, T, Frame::Inertial>.)
 */
template <size_t VolumeDim, typename Fr>
class Gaussian : public MathFunction<VolumeDim, Fr> {
 public:
  struct Amplitude {
    using type = double;
    static constexpr Options::String help = {"The amplitude."};
  };

  struct Width {
    using type = double;
    static constexpr Options::String help = {"The width."};
    static type lower_bound() noexcept { return 0.; }
  };

  struct Center {
    using type = std::array<double, VolumeDim>;
    static constexpr Options::String help = {"The center."};
  };
  using options = tmpl::list<Amplitude, Width, Center>;

  static constexpr Options::String help = {
      "Computes a Gaussian about an arbitrary coordinate center with given "
      "width and amplitude"};

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(MathFunction<VolumeDim, Fr>),
                                     Gaussian);  // NOLINT

  explicit Gaussian(CkMigrateMessage* /*unused*/) noexcept {}

  Gaussian(double amplitude, double width,
           const std::array<double, VolumeDim>& center) noexcept;

  Gaussian() = default;
  ~Gaussian() override = default;
  Gaussian(const Gaussian& /*rhs*/) = delete;
  Gaussian& operator=(const Gaussian& /*rhs*/) = delete;
  Gaussian(Gaussian&& /*rhs*/) noexcept = default;
  Gaussian& operator=(Gaussian&& /*rhs*/) noexcept = default;

  Scalar<double> operator()(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept override;
  Scalar<DataVector> operator()(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept override;

  tnsr::i<double, VolumeDim, Fr> first_deriv(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept override;
  tnsr::i<DataVector, VolumeDim, Fr> first_deriv(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept override;

  tnsr::ii<double, VolumeDim, Fr> second_deriv(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept override;
  tnsr::ii<DataVector, VolumeDim, Fr> second_deriv(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept override;

  tnsr::iii<double, VolumeDim, Fr> third_deriv(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept override;
  tnsr::iii<DataVector, VolumeDim, Fr> third_deriv(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept override;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) override;  // NOLINT

 private:
  double amplitude_{};
  double inverse_width_{};
  std::array<double, VolumeDim> center_{};
  friend bool operator==(const Gaussian& lhs, const Gaussian& rhs) noexcept {
    return lhs.amplitude_ == rhs.amplitude_ and
           lhs.inverse_width_ == rhs.inverse_width_ and
           lhs.center_ == rhs.center_;
  }

  template <typename T>
  tnsr::I<T, VolumeDim, Fr> centered_coordinates(
      const tnsr::I<T, VolumeDim, Fr>& x) const noexcept;
  template <typename T>
  Scalar<T> apply_call_operator(
      const tnsr::I<T, VolumeDim, Fr>& centered_coords) const noexcept;
  template <typename T>
  tnsr::i<T, VolumeDim, Fr> apply_first_deriv(
      const tnsr::I<T, VolumeDim, Fr>& centered_coords,
      const Scalar<T>& gaussian) const noexcept;
  template <typename T>
  tnsr::ii<T, VolumeDim, Fr> apply_second_deriv(
      const tnsr::I<T, VolumeDim, Fr>& centered_coords,
      const Scalar<T>& gaussian,
      const tnsr::i<T, VolumeDim, Fr>& d_gaussian) const noexcept;
  template <typename T>
  tnsr::iii<T, VolumeDim, Fr> apply_third_deriv(
      const tnsr::I<T, VolumeDim, Fr>& centered_coords,
      const Scalar<T>& gaussian, const tnsr::i<T, VolumeDim, Fr>& d_gaussian,
      const tnsr::ii<T, VolumeDim, Fr>& d2_gaussian) const noexcept;
};

template <size_t VolumeDim, typename Fr>
bool operator!=(const Gaussian<VolumeDim, Fr>& lhs,
                const Gaussian<VolumeDim, Fr>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace MathFunctions

/// \cond
template <size_t VolumeDim, typename Fr>
PUP::able::PUP_ID MathFunctions::Gaussian<VolumeDim, Fr>::my_PUP_ID =
    0;  // NOLINT

template <typename Fr>
PUP::able::PUP_ID MathFunctions::Gaussian<1, Fr>::my_PUP_ID = 0;  // NOLINT
/// \endcond
