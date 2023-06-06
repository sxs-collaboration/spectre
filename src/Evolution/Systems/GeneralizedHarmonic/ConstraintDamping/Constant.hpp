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

namespace gh::ConstraintDamping {
/*!
 * \brief A constant function: \f$f = C\f$
 *
 * \details Input file options are: `Value` \f$C\f$. The function takes input
 * coordinates of type `tnsr::I<T, VolumeDim, Fr>`, where `T` is e.g. `double`
 * or `DataVector`, `Fr` is a frame (e.g. `Frame::Inertial`), and `VolumeDim` is
 * the dimension of the spatial volume.
 */
template <size_t VolumeDim, typename Fr>
class Constant : public DampingFunction<VolumeDim, Fr> {
 public:
  struct Value {
    using type = double;
    static constexpr Options::String help = {"The value."};
  };
  using options = tmpl::list<Value>;

  static constexpr Options::String help = {"Returns a constant value"};

  /// \cond
  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(DampingFunction<VolumeDim, Fr>),
                                     Constant);  // NOLINT

  explicit Constant(CkMigrateMessage* msg);
  /// \endcond

  Constant(double value);

  Constant() = default;
  ~Constant() override = default;
  Constant(const Constant& /*rhs*/) = default;
  Constant& operator=(const Constant& /*rhs*/) = default;
  Constant(Constant&& /*rhs*/) = default;
  Constant& operator=(Constant&& /*rhs*/) = default;

  void operator()(const gsl::not_null<Scalar<double>*> value_at_x,
                  const tnsr::I<double, VolumeDim, Fr>& x, double time,
                  const std::unordered_map<
                      std::string,
                      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
                      functions_of_time) const override;
  void operator()(const gsl::not_null<Scalar<DataVector>*> value_at_x,
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
  friend bool operator==(const Constant& lhs, const Constant& rhs) {
    return lhs.value_ == rhs.value_;
  }

  template <typename T>
  void apply_call_operator(const gsl::not_null<Scalar<T>*> value_at_x) const;

  double value_ = std::numeric_limits<double>::signaling_NaN();
};

template <size_t VolumeDim, typename Fr>
bool operator!=(const Constant<VolumeDim, Fr>& lhs,
                const Constant<VolumeDim, Fr>& rhs) {
  return not(lhs == rhs);
}
}  // namespace gh::ConstraintDamping

/// \cond
template <size_t VolumeDim, typename Fr>
PUP::able::PUP_ID gh::ConstraintDamping::Constant<VolumeDim, Fr>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
