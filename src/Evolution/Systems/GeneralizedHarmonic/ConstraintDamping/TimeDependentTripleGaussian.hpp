// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/DampingFunction.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
/// \endcond

namespace GeneralizedHarmonic::ConstraintDamping {
/*!
 * \brief A sum of three Gaussians plus a constant, where the Gaussian widths
 * are scaled by a domain::FunctionsOfTime::FunctionOfTime.
 *
 * \details The function \f$f\f$ is given by
 * \f{align}{
 * f = C + \sum_{\alpha=1}^3
 * A_\alpha \exp\left(-\frac{(x-(x_0)_\alpha)^2}{w_\alpha^2(t)}\right).
 * \f}
 * Input file options are: `Constant` \f$C\f$, `Amplitude[1-3]`
 * \f$A_\alpha\f$, `Width[1-3]` \f$w_\alpha\f$, `Center[1-3]
 * `\f$(x_0)_\alpha\f$, and `FunctionOfTimeForScaling`, a string naming a
 * domain::FunctionsOfTime::FunctionOfTime in the domain::Tags::FunctionsOfTime
 * that will be passed to the call operator. The function takes input
 * coordinates \f$x\f$ of type `tnsr::I<T, 3, Frame::Grid>`, where `T` is e.g.
 * `double` or `DataVector`; note that this DampingFunction is only defined
 * for three spatial dimensions and for the grid frame. The Gaussian widths
 * \f$w_\alpha\f$ are scaled by the inverse of the value of a scalar
 * domain::FunctionsOfTime::FunctionOfTime \f$f(t)\f$ named
 * `FunctionOfTimeForScaling`: \f$w_\alpha(t) = w_\alpha / f(t)\f$.
 */
class TimeDependentTripleGaussian : public DampingFunction<3, Frame::Grid> {
 public:
  template <size_t GaussianNumber>
  struct Gaussian {
    static constexpr Options::String help = {
        "Parameters for one of the Gaussians."};
    static std::string name() {
      return "Gaussian" + std::to_string(GaussianNumber);
    };
  };
  struct Constant {
    using type = double;
    static constexpr Options::String help = {"The constant."};
  };

  template <typename Group>
  struct Amplitude {
    using group = Group;
    using type = double;
    static constexpr Options::String help = {"The amplitude of the Gaussian."};
  };

  template <typename Group>
  struct Width {
    using group = Group;
    using type = double;
    static constexpr Options::String help = {
        "The unscaled width of the Gaussian."};
    static type lower_bound() { return 0.; }
  };

  template <typename Group>
  struct Center {
    using group = Group;
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"The center of the Gaussian."};
  };

  struct FunctionOfTimeForScaling {
    using type = std::string;
    static constexpr Options::String help = {"The name of the FunctionOfTime."};
  };

  using options =
      tmpl::list<Constant, Amplitude<Gaussian<1>>, Width<Gaussian<1>>,
                 Center<Gaussian<1>>, Amplitude<Gaussian<2>>,
                 Width<Gaussian<2>>, Center<Gaussian<2>>,
                 Amplitude<Gaussian<3>>, Width<Gaussian<3>>,
                 Center<Gaussian<3>>, FunctionOfTimeForScaling>;

  static constexpr Options::String help = {
      "Computes a sum of a constant and 3 Gaussians (each with its own "
      "amplitude, width, and coordinate center), with the Gaussian widths "
      "scaled by the inverse of a FunctionOfTime."};

  /// \cond
  WRAPPED_PUPable_decl_base_template(
      SINGLE_ARG(DampingFunction<3, Frame::Grid>),
      TimeDependentTripleGaussian);  // NOLINT
  /// \endcond

  explicit TimeDependentTripleGaussian(CkMigrateMessage* msg);

  TimeDependentTripleGaussian(double constant, double amplitude_1,
                              double width_1,
                              const std::array<double, 3>& center_1,
                              double amplitude_2, double width_2,
                              const std::array<double, 3>& center_2,
                              double amplitude_3, double width_3,
                              const std::array<double, 3>& center_3,
                              std::string function_of_time_for_scaling);

  TimeDependentTripleGaussian() = default;
  ~TimeDependentTripleGaussian() override = default;
  TimeDependentTripleGaussian(const TimeDependentTripleGaussian& /*rhs*/) =
      default;
  TimeDependentTripleGaussian& operator=(
      const TimeDependentTripleGaussian& /*rhs*/) = default;
  TimeDependentTripleGaussian(TimeDependentTripleGaussian&& /*rhs*/) = default;
  TimeDependentTripleGaussian& operator=(
      TimeDependentTripleGaussian&& /*rhs*/) = default;

  void operator()(const gsl::not_null<Scalar<double>*> value_at_x,
                  const tnsr::I<double, 3, Frame::Grid>& x, double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time) const override;
  void operator()(const gsl::not_null<Scalar<DataVector>*> value_at_x,
                  const tnsr::I<DataVector, 3, Frame::Grid>& x, double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time) const override;

  auto get_clone() const
      -> std::unique_ptr<DampingFunction<3, Frame::Grid>> override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const TimeDependentTripleGaussian& lhs,
                         const TimeDependentTripleGaussian& rhs) {
    return lhs.constant_ == rhs.constant_ and
           lhs.amplitude_1_ == rhs.amplitude_1_ and
           lhs.inverse_width_1_ == rhs.inverse_width_1_ and
           lhs.center_1_ == rhs.center_1_ and
           lhs.amplitude_2_ == rhs.amplitude_2_ and
           lhs.inverse_width_2_ == rhs.inverse_width_2_ and
           lhs.center_2_ == rhs.center_2_ and
           lhs.amplitude_3_ == rhs.amplitude_3_ and
           lhs.inverse_width_3_ == rhs.inverse_width_3_ and
           lhs.center_3_ == rhs.center_3_ and
           lhs.function_of_time_for_scaling_ ==
               rhs.function_of_time_for_scaling_;
  }

  double constant_ = std::numeric_limits<double>::signaling_NaN();
  double amplitude_1_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_width_1_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> center_1_{};
  double amplitude_2_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_width_2_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> center_2_{};
  double amplitude_3_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_width_3_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> center_3_{};
  std::string function_of_time_for_scaling_;

  template <typename T>
  void apply_call_operator(
      const gsl::not_null<Scalar<T>*> value_at_x,
      const tnsr::I<T, 3, Frame::Grid>& x, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;
};
}  // namespace GeneralizedHarmonic::ConstraintDamping
