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
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/ConstraintDamping/DampingFunction.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
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

namespace ConstraintDamping {
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
 * \f$A_\alpha\f$, `Width[1-3]` \f$w_\alpha\f$, and `Center[1-3]
 * `\f$(x_0)_\alpha\f$. The function takes input
 * coordinates \f$x\f$ of type `tnsr::I<T, 3, Frame::Grid>`, where `T` is e.g.
 * `double` or `DataVector`; note that this DampingFunction is only defined
 * for three spatial dimensions and for the grid frame. The Gaussian widths
 * \f$w_\alpha\f$ are scaled by the inverse of the value of a scalar
 * domain::FunctionsOfTime::FunctionOfTime \f$f(t)\f$: \f$w_\alpha(t) = w_\alpha
 * / f(t)\f$.
 *
 * You can choose one of two methods for tracking the object
 * centers. `ExpansionFactor` should be used for BBH simulations where the
 * expansion control system is used to track the objects. `ObjectCenters`
 * sholud be used for BNS simulations where the coordinate centers of the two
 * stars is tracked separately and there is no expansion control system.
 */
class TimeDependentTripleGaussian : public DampingFunction<3, Frame::Grid> {
 private:
  enum class MovementMethods { ExpansionFactor, ObjectCenters };

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
    using type = tmpl::conditional_t<std::is_same_v<Group, Gaussian<3>>,
                                     std::array<double, 3>,
                                     Options::Auto<std::array<double, 3>>>;
    static constexpr Options::String help = {"The center of the Gaussian."};
  };

  /// \brief How to track the movement of the compact objects.
  ///
  /// - `ExpansionFactor` for BBH simulations.
  /// - `ObjectCenters` for BNS simulations.
  struct MovementMethod {
    using type = std::string;
    static constexpr Options::String help = {
        "How to track the movement of the compact objects.\n\n"
        "- `ExpansionFactor` for BBH simulations.\n"
        "- `ObjectCenters` for BNS simulations."};
  };

  using options =
      tmpl::list<Constant, Amplitude<Gaussian<1>>, Width<Gaussian<1>>,
                 Center<Gaussian<1>>, Amplitude<Gaussian<2>>,
                 Width<Gaussian<2>>, Center<Gaussian<2>>,
                 Amplitude<Gaussian<3>>, Width<Gaussian<3>>,
                 Center<Gaussian<3>>, MovementMethod>;

  static constexpr Options::String help = {
      "Computes a sum of a constant and 3 Gaussians (each with its own "
      "amplitude, width, and coordinate center), with the Gaussian widths "
      "scaled by the inverse of a FunctionOfTime."};

  /// \cond
  WRAPPED_PUPable_decl_base_template(
      SINGLE_ARG(DampingFunction<3, Frame::Grid>),
      TimeDependentTripleGaussian);  // NOLINT
  /// \endcond

  TimeDependentTripleGaussian(
      double constant, double amplitude_1, double width_1,
      const std::optional<std::array<double, 3>>& center_1, double amplitude_2,
      double width_2, const std::optional<std::array<double, 3>>& center_2,
      double amplitude_3, double width_3, const std::array<double, 3>& center_3,
      const std::string& movement_method, const Options::Context& context = {});

  TimeDependentTripleGaussian() = default;
  ~TimeDependentTripleGaussian() override = default;
  TimeDependentTripleGaussian(const TimeDependentTripleGaussian& /*rhs*/) =
      default;
  TimeDependentTripleGaussian& operator=(
      const TimeDependentTripleGaussian& /*rhs*/) = default;
  TimeDependentTripleGaussian(TimeDependentTripleGaussian&& /*rhs*/) = default;
  TimeDependentTripleGaussian& operator=(
      TimeDependentTripleGaussian&& /*rhs*/) = default;

  void operator()(
      gsl::not_null<Scalar<double>*> value_at_x,
      const tnsr::I<double, 3, Frame::Grid>& x, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const override;
  void operator()(
      gsl::not_null<Scalar<DataVector>*> value_at_x,
      const tnsr::I<DataVector, 3, Frame::Grid>& x, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const override;

  auto get_clone() const
      -> std::unique_ptr<DampingFunction<3, Frame::Grid>> override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const TimeDependentTripleGaussian& lhs,
                         const TimeDependentTripleGaussian& rhs);

  double constant_ = std::numeric_limits<double>::signaling_NaN();
  double amplitude_1_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_width_1_ = std::numeric_limits<double>::signaling_NaN();
  std::optional<std::array<double, 3>> center_1_{};
  double amplitude_2_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_width_2_ = std::numeric_limits<double>::signaling_NaN();
  std::optional<std::array<double, 3>> center_2_{};
  double amplitude_3_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_width_3_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> center_3_{};
  MovementMethods movement_method_{MovementMethods::ExpansionFactor};
  inline static const std::string function_of_time_for_scaling_{"Expansion"};
  inline static const std::string function_of_time_for_centers_{"GridCenters"};

  template <typename T>
  void apply_call_operator(
      gsl::not_null<Scalar<T>*> value_at_x, const tnsr::I<T, 3, Frame::Grid>& x,
      double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;
};

bool operator!=(const TimeDependentTripleGaussian& lhs,
                const TimeDependentTripleGaussian& rhs);
}  // namespace ConstraintDamping
