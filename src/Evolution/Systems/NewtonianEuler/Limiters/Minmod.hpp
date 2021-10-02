// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstdlib>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tags.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/VariablesToLimit.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class Element;
template <size_t VolumeDim>
class ElementId;
template <size_t VolumeDim>
class Mesh;
template <size_t VolumeDim>
class OrientationMap;

namespace boost {
template <class T>
struct hash;
}  // namespace boost

namespace PUP {
class er;
}  // namespace PUP

namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
template <size_t VolumeDim>
struct Element;
template <size_t VolumeDim>
struct Mesh;
template <size_t VolumeDim>
struct SizeOfElement;
}  // namespace Tags
}  // namespace domain
/// \endcond

namespace NewtonianEuler {
namespace Limiters {

/// \ingroup LimitersGroup
/// \brief A minmod-based generalized slope limiter for the NewtonianEuler
/// system.
///
/// Implements the three minmod-based generalized slope limiters from
/// \cite Cockburn1999 Sec. 2.4: \f$\Lambda\Pi^1\f$, \f$\Lambda\Pi^N\f$, and
/// MUSCL. See the documentation of the system-agnostic ::Limiters::Minmod
/// limiter for a general discussion of the algorithm and the various options
/// that control the action of the limiter.
///
/// This implemention is specialized to the NewtonianEuler evolution system.
/// By specializing the limiter to the system, we can add two features that
/// improve its robustness:
/// - the limiter can be applied to the system's characteristic variables. This
///   is the recommendation of the reference, because it reduces spurious
///   oscillations in the post-limiter solution.
/// - after limiting, the solution can be processed to remove any remaining
///   unphysical values like negative densities and pressures. We do this by
///   scaling the solution around its mean (a "flattener" or "bounds-preserving"
///   filter). Note: the flattener is applied to all elements, including those
///   where the limiter did not act to reduce the solution's slopes.
template <size_t VolumeDim>
class Minmod {
 public:
  using ConservativeVarsMinmod = ::Limiters::Minmod<
      VolumeDim, tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                            NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                            NewtonianEuler::Tags::EnergyDensity>>;

  struct VariablesToLimit {
    using type = NewtonianEuler::Limiters::VariablesToLimit;
    static type suggested_value() {
      return NewtonianEuler::Limiters::VariablesToLimit::Characteristic;
    }
    static constexpr Options::String help = {
        "Variable representation on which to apply the limiter"};
  };
  struct ApplyFlattener {
    using type = bool;
    static constexpr Options::String help = {
        "Flatten after limiting to restore pointwise positivity"};
  };
  using options =
      tmpl::list<typename ConservativeVarsMinmod::Type, VariablesToLimit,
                 typename ConservativeVarsMinmod::TvbConstant, ApplyFlattener,
                 typename ConservativeVarsMinmod::DisableForDebugging>;
  static constexpr Options::String help = {
      "A Minmod limiter specialized to the NewtonianEuler system"};
  static std::string name() { return "NewtonianEulerMinmod"; };

  explicit Minmod(::Limiters::MinmodType minmod_type,
                  NewtonianEuler::Limiters::VariablesToLimit vars_to_limit,
                  double tvb_constant, bool apply_flattener,
                  bool disable_for_debugging = false);

  Minmod() = default;
  Minmod(const Minmod& /*rhs*/) = default;
  Minmod& operator=(const Minmod& /*rhs*/) = default;
  Minmod(Minmod&& /*rhs*/) = default;
  Minmod& operator=(Minmod&& /*rhs*/) = default;
  ~Minmod() = default;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p);  // NOLINT

  using PackagedData = typename ConservativeVarsMinmod::PackagedData;
  using package_argument_tags =
      typename ConservativeVarsMinmod::package_argument_tags;

  /// \brief Package data for sending to neighbor elements.
  void package_data(gsl::not_null<PackagedData*> packaged_data,
                    const Scalar<DataVector>& mass_density_cons,
                    const tnsr::I<DataVector, VolumeDim>& momentum_density,
                    const Scalar<DataVector>& energy_density,
                    const Mesh<VolumeDim>& mesh,
                    const std::array<double, VolumeDim>& element_size,
                    const OrientationMap<VolumeDim>& orientation_map) const;

  using limit_tags =
      tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                 NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                 NewtonianEuler::Tags::EnergyDensity>;
  using limit_argument_tags = tmpl::list<
      domain::Tags::Mesh<VolumeDim>, domain::Tags::Element<VolumeDim>,
      domain::Tags::Coordinates<VolumeDim, Frame::ElementLogical>,
      domain::Tags::SizeOfElement<VolumeDim>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      ::hydro::Tags::EquationOfStateBase>;

  /// \brief Limits the solution on the element.
  template <size_t ThermodynamicDim>
  bool operator()(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
      const tnsr::I<DataVector, VolumeDim, Frame::ElementLogical>&
          logical_coords,
      const std::array<double, VolumeDim>& element_size,
      const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state,
      const std::unordered_map<
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
          neighbor_data) const;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const Minmod<LocalDim>& lhs,
                         const Minmod<LocalDim>& rhs);

  ::Limiters::MinmodType minmod_type_;
  NewtonianEuler::Limiters::VariablesToLimit vars_to_limit_;
  double tvb_constant_;
  bool apply_flattener_;
  bool disable_for_debugging_;
  ConservativeVarsMinmod conservative_vars_minmod_;
};

template <size_t VolumeDim>
bool operator!=(const Minmod<VolumeDim>& lhs, const Minmod<VolumeDim>& rhs);

}  // namespace Limiters
}  // namespace NewtonianEuler
