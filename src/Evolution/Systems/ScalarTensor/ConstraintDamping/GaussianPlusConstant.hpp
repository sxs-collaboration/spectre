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
#include "Evolution/Systems/ScalarTensor/ConstraintDamping/DampingFunction.hpp"
#include "Options/String.hpp"
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

namespace ScalarTensor::ConstraintDamping {
/*!
 * \brief A Gaussian plus a constant: \f$f = C + A
 * \exp\left(-\frac{(x-x_0)^2}{w^2}\right)\f$
 *
 * \details Input file options are: `Constant` \f$C\f$, `Amplitude` \f$A\f$,
 * `Width` \f$w\f$, and `Center`\f$x_0\f$. The function takes input coordinates
 * of type `tnsr::I<T, VolumeDim, Fr>`, where `T` is e.g. `double` or
 * `DataVector`, `Fr` is a frame (e.g. `Frame::Inertial`), and `VolumeDim` is
 * the dimension of the spatial volume.
 */
template <size_t VolumeDim, typename Fr>
class GaussianPlusConstant : public DampingFunction<VolumeDim, Fr> {
 public:
  struct Constant {
    using type = double;
    static constexpr Options::String help = {"The constant."};
  };

  struct Amplitude {
    using type = double;
    static constexpr Options::String help = {"The amplitude of the Gaussian."};
  };

  struct Width {
    using type = double;
    static constexpr Options::String help = {"The width of the Gaussian."};
    static type lower_bound() { return 0.; }
  };

  struct Center {
    using type = std::array<double, VolumeDim>;
    static constexpr Options::String help = {"The center of the Gaussian."};
  };
  using options = tmpl::list<Constant, Amplitude, Width, Center>;

  static constexpr Options::String help = {
      "Computes a Gaussian plus a constant about an arbitrary coordinate "
      "center with given width and amplitude"};

  /// \cond
  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(DampingFunction<VolumeDim, Fr>),
                                     GaussianPlusConstant);  // NOLINT

  explicit GaussianPlusConstant(CkMigrateMessage* msg);
  /// \endcond

  GaussianPlusConstant(double constant, double amplitude, double width,
                       const std::array<double, VolumeDim>& center);

  GaussianPlusConstant() = default;
  ~GaussianPlusConstant() override = default;
  GaussianPlusConstant(const GaussianPlusConstant& /*rhs*/) = default;
  GaussianPlusConstant& operator=(const GaussianPlusConstant& /*rhs*/) =
      default;
  GaussianPlusConstant(GaussianPlusConstant&& /*rhs*/) = default;
  GaussianPlusConstant& operator=(GaussianPlusConstant&& /*rhs*/) = default;

  void operator()(gsl::not_null<Scalar<double>*> value_at_x,
                  const tnsr::I<double, VolumeDim, Fr>& x, double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time) const override;
  void operator()(gsl::not_null<Scalar<DataVector>*> value_at_x,
                  const tnsr::I<DataVector, VolumeDim, Fr>& x, double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time) const override;

  auto get_clone() const
      -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const GaussianPlusConstant& lhs,
                         const GaussianPlusConstant& rhs) {
    return lhs.constant_ == rhs.constant_ and
           lhs.amplitude_ == rhs.amplitude_ and
           lhs.inverse_width_ == rhs.inverse_width_ and
           lhs.center_ == rhs.center_;
  }

  double constant_ = std::numeric_limits<double>::signaling_NaN();
  double amplitude_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_width_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, VolumeDim> center_{};

  template <typename T>
  void apply_call_operator(gsl::not_null<Scalar<T>*> value_at_x,
                           const tnsr::I<T, VolumeDim, Fr>& x) const;
};

template <size_t VolumeDim, typename Fr>
bool operator!=(const GaussianPlusConstant<VolumeDim, Fr>& lhs,
                const GaussianPlusConstant<VolumeDim, Fr>& rhs) {
  return not(lhs == rhs);
}
}  // namespace ScalarTensor::ConstraintDamping

/// \cond
template <size_t VolumeDim, typename Fr>
PUP::able::PUP_ID ScalarTensor::ConstraintDamping::GaussianPlusConstant<
    VolumeDim, Fr>::my_PUP_ID = 0;  // NOLINT
/// \endcond
