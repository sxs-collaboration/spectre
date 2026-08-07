// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace SecondOrderScalarWave::Tags {
struct Psi;
struct Pi;
template <size_t Dim>
struct Phi;
}  // namespace SecondOrderScalarWave::Tags
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace SecondOrderScalarWave::Solutions {
/*!
 * \brief Adapts a `ScalarWave` analytic solution to an analytic solution of
 * the `SecondOrderScalarWave` system.
 *
 * The wrapped `SolutionType` supplies the mathematics of the solution together
 * with its variables and their time derivatives in the `ScalarWave` tag
 * namespace. This adapter re-exposes those quantities through the
 * `SecondOrderScalarWave` tags, adding the three `variables` interfaces the
 * second-order-in-space system requires: the evolved and auxiliary variables
 * \f$\{\Psi, \Pi, \Phi_i\}\f$, the evolved variables \f$\{\Psi, \Pi\}\f$, and
 * the time derivatives of the evolved variables
 * \f$\{\partial_t\Psi, \partial_t\Pi\}\f$. In the second-order-in-space system
 * only \f$\Psi\f$ and \f$\Pi\f$ are evolved; \f$\Phi_i\f$ is auxiliary and is
 * not evolved, so \f$\partial_t\Phi_i\f$ from the wrapped solution is
 * discarded.
 *
 * The wrapped solution is held by composition, so only the second-order
 * interface below is public; the `ScalarWave` interface of `SolutionType` is
 * not exposed.
 *
 * \tparam SolutionType a `ScalarWave` analytic solution to adapt
 */
template <typename SolutionType>
class SecondOrderWrapper : public evolution::initial_data::InitialData,
                           public MarkAsAnalyticSolution {
 private:
  // Lifetime-safe backing storage for `help`, built on first use so it is
  // constructed before `help` reads it.
  static const std::string& help_storage() {
    static const std::string storage =
        "Adapts a ScalarWave analytic solution for SecondOrderScalarWave.\n" +
        std::string{SolutionType::help};
    return storage;
  }

 public:
  SecondOrderWrapper() = default;
  SecondOrderWrapper(const SecondOrderWrapper& /*rhs*/) = default;
  SecondOrderWrapper& operator=(const SecondOrderWrapper& /*rhs*/) = default;
  SecondOrderWrapper(SecondOrderWrapper&& /*rhs*/) = default;
  SecondOrderWrapper& operator=(SecondOrderWrapper&& /*rhs*/) = default;
  ~SecondOrderWrapper() override = default;

  explicit SecondOrderWrapper(SolutionType solution)
      : wrapped_solution_(std::move(solution)) {}

  template <typename... Args>
  requires std::is_constructible_v<SolutionType, Args&&...>
  explicit SecondOrderWrapper(Args&&... args)
      : wrapped_solution_(std::forward<Args>(args)...) {}

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit SecondOrderWrapper(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SecondOrderWrapper);
  /// \endcond

  static constexpr size_t volume_dim = SolutionType::volume_dim;
  using options = typename SolutionType::options;
  inline static Options::String help = help_storage().c_str();
  static std::string name() {
    return "SecondOrder" + pretty_type::name<SolutionType>();
  }

  using tags = tmpl::list<SecondOrderScalarWave::Tags::Psi,
                          SecondOrderScalarWave::Tags::Pi,
                          SecondOrderScalarWave::Tags::Phi<volume_dim>,
                          ::Tags::dt<SecondOrderScalarWave::Tags::Psi>,
                          ::Tags::dt<SecondOrderScalarWave::Tags::Pi>>;

  /// Retrieve the evolved and auxiliary variables (Psi, Pi, Phi) at time `t`
  /// and spatial coordinates `x`
  tuples::TaggedTuple<SecondOrderScalarWave::Tags::Psi,
                      SecondOrderScalarWave::Tags::Pi,
                      SecondOrderScalarWave::Tags::Phi<volume_dim>>
  variables(const tnsr::I<DataVector, volume_dim>& x, double t,
            tmpl::list<SecondOrderScalarWave::Tags::Psi,
                       SecondOrderScalarWave::Tags::Pi,
                       SecondOrderScalarWave::Tags::Phi<volume_dim>>
            /*meta*/) const;

  /// Retrieve the evolved variables (Psi, Pi)
  tuples::TaggedTuple<SecondOrderScalarWave::Tags::Psi,
                      SecondOrderScalarWave::Tags::Pi>
  variables(const tnsr::I<DataVector, volume_dim>& x, double t,
            tmpl::list<SecondOrderScalarWave::Tags::Psi,
                       SecondOrderScalarWave::Tags::Pi> /*meta*/) const;

  /// Retrieve the time derivatives of the evolved variables (dt(Psi), dt(Pi))
  /// at time `t` and spatial coordinates `x`
  tuples::TaggedTuple<::Tags::dt<SecondOrderScalarWave::Tags::Psi>,
                      ::Tags::dt<SecondOrderScalarWave::Tags::Pi>>
  variables(const tnsr::I<DataVector, volume_dim>& x, double t,
            tmpl::list<::Tags::dt<SecondOrderScalarWave::Tags::Psi>,
                       ::Tags::dt<SecondOrderScalarWave::Tags::Pi>>
            /*meta*/) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const SecondOrderWrapper& lhs,
                         const SecondOrderWrapper& rhs) {
    return lhs.wrapped_solution_ == rhs.wrapped_solution_;
  }
  friend bool operator!=(const SecondOrderWrapper& lhs,
                         const SecondOrderWrapper& rhs) {
    return not(lhs == rhs);
  }

  SolutionType wrapped_solution_{};
};
}  // namespace SecondOrderScalarWave::Solutions
