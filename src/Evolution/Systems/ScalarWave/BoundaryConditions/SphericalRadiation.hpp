// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
namespace Options {
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace ScalarWave::BoundaryConditions {
namespace detail {
/// The type of spherical radiation boundary condition to impose
enum class SphericalRadiationType {
  /// Impose \f$(\partial_t + \partial_r)\Psi=0\f$
  Sommerfeld,
  /// Impose \f$(\partial_t + \partial_r + r^{-1})\Psi=0\f$
  BaylissTurkel
};

SphericalRadiationType convert_spherical_radiation_type_from_yaml(
    const Options::Option& options);
}  // namespace detail

/*!
 * \brief Impose spherical radiation boundary conditions.
 *
 * These can be imposed in one of two forms:
 *
 * \f{align*}{
 *  \Pi=\partial_r\Psi,
 * \f}
 *
 * referred to as a Sommerfeld condition, or
 *
 * \f{align*}{
 *  \Pi=\partial_r\Psi + \frac{1}{r}\Psi
 * \f}
 *
 * referred to as a Bayliss-Turkel condition.
 *
 * The Bayliss-Turkel condition produces fewer reflections than the Sommerfeld
 * condition.
 *
 * \warning These are implemented assuming the outer boundary is spherical and
 * centered at the origin of the radiation because the code sets
 * \f$\Pi=n^i\Phi_i\f$, where \f$n^i\f$ is the outward pointing unit normal
 * vector. It might be possible to generalize the condition to non-spherical
 * boundaries by using \f$x^i/r\f$ instead of \f$n^i\f$, but this hasn't been
 * tested.
 */
template <size_t Dim>
class SphericalRadiation final : public BoundaryCondition<Dim> {
 public:
  struct TypeOptionTag {
    using type = detail::SphericalRadiationType;
    static std::string name() noexcept { return "Type"; }
    static constexpr Options::String help{
        "Whether to impose Sommerfeld or first-order Bayliss-Turkel spherical "
        "radiation boundary conditions."};
  };

  using options = tmpl::list<TypeOptionTag>;
  static constexpr Options::String help{
      "Spherical radiation boundary conditions setting the value of Psi, Phi, "
      "and Pi either using the Sommerfeld or first-order Bayliss-Turkel "
      "method."};

  SphericalRadiation() = default;
  SphericalRadiation(detail::SphericalRadiationType type) noexcept;
  SphericalRadiation(SphericalRadiation&&) noexcept = default;
  SphericalRadiation& operator=(SphericalRadiation&&) noexcept = default;
  SphericalRadiation(const SphericalRadiation&) = default;
  SphericalRadiation& operator=(const SphericalRadiation&) = default;
  ~SphericalRadiation() override = default;

  explicit SphericalRadiation(CkMigrateMessage* msg) noexcept;

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, SphericalRadiation);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<Phi<Dim>, Psi>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 Tags::ConstraintGamma2>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_ghost(
      gsl::not_null<Scalar<DataVector>*> pi_ext,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi_ext,
      gsl::not_null<Scalar<DataVector>*> psi_ext,
      gsl::not_null<Scalar<DataVector>*> gamma2_ext,
      const std::optional<
          tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
      const Scalar<DataVector>& psi,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
      const Scalar<DataVector>& gamma2) const noexcept;

 private:
  detail::SphericalRadiationType type_{
      detail::SphericalRadiationType::Sommerfeld};
};
}  // namespace ScalarWave::BoundaryConditions

template <>
struct Options::create_from_yaml<
    ScalarWave::BoundaryConditions::detail::SphericalRadiationType> {
  template <typename Metavariables>
  static typename ScalarWave::BoundaryConditions::detail::SphericalRadiationType
  create(const Options::Option& options) {
    return ScalarWave::BoundaryConditions::detail::
        convert_spherical_radiation_type_from_yaml(options);
  }
};
