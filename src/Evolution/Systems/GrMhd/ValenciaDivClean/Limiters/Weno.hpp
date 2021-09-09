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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/VariablesToLimit.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
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

namespace grmhd::ValenciaDivClean::Limiters {

/// \ingroup LimitersGroup
/// \brief A compact-stencil WENO limiter for the ValenciaDivClean system.
///
/// An in-progress experiment to use characteristic limiting with GRMHD...
class Weno {
 public:
  using ConservativeVarsWeno =
      ::Limiters::Weno<3, tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                                     grmhd::ValenciaDivClean::Tags::TildeTau,
                                     grmhd::ValenciaDivClean::Tags::TildeS<>,
                                     grmhd::ValenciaDivClean::Tags::TildeB<>,
                                     grmhd::ValenciaDivClean::Tags::TildePhi>>;

  struct VariablesToLimit {
    using type = grmhd::ValenciaDivClean::Limiters::VariablesToLimit;
    static type suggested_value() noexcept {
      return grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic;
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
      "A WENO limiter specialized to the ValenciaDivClean system"};
  static std::string name() noexcept { return "ValenciaDivCleanWeno"; };

  Weno(::Limiters::WenoType weno_type,
       grmhd::ValenciaDivClean::Limiters::VariablesToLimit vars_to_limit,
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
  void package_data(gsl::not_null<PackagedData*> packaged_data,
                    const Scalar<DataVector>& tilde_d,
                    const Scalar<DataVector>& tilde_tau,
                    const tnsr::i<DataVector, 3>& tilde_s,
                    const tnsr::I<DataVector, 3>& tilde_b,
                    const Scalar<DataVector>& tilde_phi, const Mesh<3>& mesh,
                    const std::array<double, 3>& element_size,
                    const OrientationMap<3>& orientation_map) const noexcept;

  using limit_tags = tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                                grmhd::ValenciaDivClean::Tags::TildeTau,
                                grmhd::ValenciaDivClean::Tags::TildeS<>,
                                grmhd::ValenciaDivClean::Tags::TildeB<>,
                                grmhd::ValenciaDivClean::Tags::TildePhi>;
  using limit_argument_tags =
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<>, gr::Tags::Lapse<>,
                 gr::Tags::Shift<3>, gr::Tags::SpatialMetric<3>,
                 domain::Tags::Mesh<3>, domain::Tags::Element<3>,
                 domain::Tags::SizeOfElement<3>,
                 domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>,
                 evolution::dg::Tags::NormalCovectorAndMagnitude<3>,
                 ::hydro::Tags::EquationOfStateBase>;

  /// \brief Limit the solution on the element
  template <size_t ThermodynamicDim>
  bool operator()(
      gsl::not_null<Scalar<DataVector>*> tilde_d,
      gsl::not_null<Scalar<DataVector>*> tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3>*> tilde_b,
      gsl::not_null<Scalar<DataVector>*> tilde_phi,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
      const tnsr::ii<DataVector, 3>& spatial_metric, const Mesh<3>& mesh,
      const Element<3>& element, const std::array<double, 3>& element_size,
      const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
      const typename evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type&
          normals_and_magnitudes,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state,
      const std::unordered_map<
          std::pair<Direction<3>, ElementId<3>>, PackagedData,
          boost::hash<std::pair<Direction<3>, ElementId<3>>>>& neighbor_data)
      const noexcept;

 private:
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const Weno& lhs, const Weno& rhs) noexcept;

  ::Limiters::WenoType weno_type_;
  grmhd::ValenciaDivClean::Limiters::VariablesToLimit vars_to_limit_;
  double neighbor_linear_weight_;
  std::optional<double> tvb_constant_;
  std::optional<double> kxrcf_constant_;
  bool apply_flattener_;
  bool disable_for_debugging_;
  ConservativeVarsWeno conservative_vars_weno_;
};

bool operator!=(const Weno& lhs, const Weno& rhs) noexcept;

}  // namespace grmhd::ValenciaDivClean::Limiters
