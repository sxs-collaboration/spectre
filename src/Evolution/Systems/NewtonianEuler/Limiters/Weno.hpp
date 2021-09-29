// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Limiters/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/NewtonianEuler/Limiters/VariablesToLimit.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
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

namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace NewtonianEuler {
namespace Limiters {

/// \ingroup LimitersGroup
/// \brief A compact-stencil WENO limiter for the NewtonianEuler system.
///
/// Implements the simple WENO limiter of \cite Zhong2013 and the Hermite WENO
/// (HWENO) limiter of \cite Zhu2016. See the documentation of the
/// system-agnostic ::Limiters::Weno limiter for a general discussion of the
/// algorithm and the various options that control the action of the limiter.
//
/// This implemention is specialized to the NewtonianEuler evolution system.
/// By specializing the limiter to the system, we can add a few features that
/// improve its robustness:
/// - the troubled-cell indicator (TCI) can be specialized to the features of
///   the evolution system.
/// - the limiter can be applied to the system's characteristic variables. This
///   is the recommendation of the reference, because it reduces spurious
///   oscillations in the post-limiter solution.
/// - after limiting, the solution can be processed to remove any remaining
///   unphysical values like negative densities and pressures. We do this by
///   scaling the solution around its mean (a "flattener" or "bounds-preserving"
///   filter). Note: the flattener is applied to all elements, including those
///   where the limiter did not act to reduce the solution's slopes.
///
/// The matrix of TCI, variables to limit, post-processing, etc. choices can
/// rapidly grow large. Here we reduce the possibilities by tying the TCI to the
/// limiter in keeping with each limiter's main reference: HWENO uses the KXRCF
/// TCI and simple WENO uses the TVB TCI. To fully explore the matrix of
/// possibilities, the source code could be generalized --- however, experience
/// suggests it is unlikely that there exists one combination that will perform
/// remarkably better than the others.
template <size_t VolumeDim>
class Weno {
 public:
  using ConservativeVarsWeno = ::Limiters::Weno<
      VolumeDim, tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                            NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                            NewtonianEuler::Tags::EnergyDensity>>;

  struct VariablesToLimit {
    using type = NewtonianEuler::Limiters::VariablesToLimit;
    static type suggested_value() noexcept {
      return NewtonianEuler::Limiters::VariablesToLimit::Characteristic;
    }
    static constexpr Options::String help = {
        "Variable representation on which to apply the limiter"};
  };
  // Future design improvement: attach the TvbConstant/KxrcfConstant to the
  // limiter type, so that it isn't necessary to specify both (but with one
  // required to be 'None') in each input file.
  struct TvbConstant {
    using type = Options::Auto<double, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "Constant in RHS of the TVB minmod TCI, used when Type = SimpleWeno"};
  };
  struct KxrcfConstant {
    using type = Options::Auto<double, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "Constant in RHS of KXRCF TCI, used when Type = Hweno"};
  };
  struct ApplyFlattener {
    using type = bool;
    static constexpr Options::String help = {
        "Flatten after limiting to restore pointwise positivity"};
  };
  using options =
      tmpl::list<typename ConservativeVarsWeno::Type, VariablesToLimit,
                 typename ConservativeVarsWeno::NeighborWeight, TvbConstant,
                 KxrcfConstant, ApplyFlattener,
                 typename ConservativeVarsWeno::DisableForDebugging>;
  static constexpr Options::String help = {
      "A WENO limiter specialized to the NewtonianEuler system"};
  static std::string name() noexcept { return "NewtonianEulerWeno"; };

  Weno(::Limiters::WenoType weno_type,
       NewtonianEuler::Limiters::VariablesToLimit vars_to_limit,
       double neighbor_linear_weight, std::optional<double> tvb_constant,
       std::optional<double> kxrcf_constant, bool apply_flattener,
       bool disable_for_debugging = false,
       const Options::Context& context = {});

  Weno() noexcept = default;
  Weno(const Weno& /*rhs*/) = default;
  Weno& operator=(const Weno& /*rhs*/) = default;
  Weno(Weno&& /*rhs*/) noexcept = default;
  Weno& operator=(Weno&& /*rhs*/) noexcept = default;
  ~Weno() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  using PackagedData = typename ConservativeVarsWeno::PackagedData;
  using package_argument_tags =
      typename ConservativeVarsWeno::package_argument_tags;

  /// \brief Package data for sending to neighbor elements
  void package_data(
      gsl::not_null<PackagedData*> packaged_data,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, VolumeDim>& momentum_density,
      const Scalar<DataVector>& energy_density, const Mesh<VolumeDim>& mesh,
      const std::array<double, VolumeDim>& element_size,
      const OrientationMap<VolumeDim>& orientation_map) const noexcept;

  using limit_tags =
      tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                 NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                 NewtonianEuler::Tags::EnergyDensity>;
  using limit_argument_tags = tmpl::list<
      domain::Tags::Mesh<VolumeDim>, domain::Tags::Element<VolumeDim>,
      domain::Tags::SizeOfElement<VolumeDim>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<VolumeDim>,
      ::hydro::Tags::EquationOfStateBase>;

  /// \brief Limit the solution on the element
  template <size_t ThermodynamicDim>
  bool operator()(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
      const std::array<double, VolumeDim>& element_size,
      const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
      const typename evolution::dg::Tags::NormalCovectorAndMagnitude<
          VolumeDim>::type& normals_and_magnitudes,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state,
      const std::unordered_map<
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
          neighbor_data) const noexcept;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const Weno<LocalDim>& lhs,
                         const Weno<LocalDim>& rhs) noexcept;

  ::Limiters::WenoType weno_type_;
  NewtonianEuler::Limiters::VariablesToLimit vars_to_limit_;
  double neighbor_linear_weight_;
  std::optional<double> tvb_constant_;
  std::optional<double> kxrcf_constant_;
  bool apply_flattener_;
  bool disable_for_debugging_;
  // Note: conservative_vars_weno_ is always used for calls to package_data, and
  // is also used when limiting cons vars with the simple WENO algorithm. So we
  // construct conservative_vars_weno_ with the correct TVB constant when it is
  // used for limiting (precisely, when weno_type_ == SimpleWeno), and with a
  // dummy TVB constant value of NaN otherwise (weno_type_ == Hweno). This lets
  // us construct the conservative_vars_weno_ variable and use it for delegating
  // the package_data work even when the specialized limiter has no TVB constant
  // and is possible because package_data doesn't depend on the TCI.
  ConservativeVarsWeno conservative_vars_weno_;
};

template <size_t VolumeDim>
bool operator!=(const Weno<VolumeDim>& lhs,
                const Weno<VolumeDim>& rhs) noexcept;

}  // namespace Limiters
}  // namespace NewtonianEuler
