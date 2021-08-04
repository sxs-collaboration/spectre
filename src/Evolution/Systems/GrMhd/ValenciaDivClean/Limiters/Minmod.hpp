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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/VariablesToLimit.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
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

namespace grmhd::ValenciaDivClean::Limiters {

/// \ingroup LimitersGroup
/// \brief A minmod-based generalized slope limiter for the ValenciaDivClean
/// system.
///
/// An in-progress experiment to use characteristic limiting with GRMHD...
class Minmod {
 public:
  using ConservativeVarsMinmod =
      ::Limiters::Minmod<3,
                         tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
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
      "A Minmod limiter specialized to the ValenciaDivClean system"};
  static std::string name() noexcept { return "ValenciaDivCleanMinmod"; };

  explicit Minmod(
      ::Limiters::MinmodType minmod_type,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit vars_to_limit,
      double tvb_constant, bool apply_flattener,
      bool disable_for_debugging = false) noexcept;

  Minmod() noexcept = default;
  Minmod(const Minmod& /*rhs*/) = default;
  Minmod& operator=(const Minmod& /*rhs*/) = default;
  Minmod(Minmod&& /*rhs*/) noexcept = default;
  Minmod& operator=(Minmod&& /*rhs*/) noexcept = default;
  ~Minmod() = default;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  using PackagedData = typename ConservativeVarsMinmod::PackagedData;
  using package_argument_tags =
      typename ConservativeVarsMinmod::package_argument_tags;

  /// \brief Package data for sending to neighbor elements.
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
                 domain::Tags::Coordinates<3, Frame::Logical>,
                 domain::Tags::SizeOfElement<3>,
                 domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>,
                 ::hydro::Tags::EquationOfStateBase>;

  /// \brief Limits the solution on the element.
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
      const Element<3>& element,
      const tnsr::I<DataVector, 3, Frame::Logical>& logical_coords,
      const std::array<double, 3>& element_size,
      const Scalar<DataVector>& det_inv_logical_to_inertial_jacobian,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state,
      const std::unordered_map<
          std::pair<Direction<3>, ElementId<3>>, PackagedData,
          boost::hash<std::pair<Direction<3>, ElementId<3>>>>& neighbor_data)
      const noexcept;

 private:
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const Minmod& lhs, const Minmod& rhs) noexcept;

  ::Limiters::MinmodType minmod_type_;
  grmhd::ValenciaDivClean::Limiters::VariablesToLimit vars_to_limit_;
  double tvb_constant_;
  bool apply_flattener_;
  bool disable_for_debugging_;
  ConservativeVarsMinmod conservative_vars_minmod_;
};

bool operator!=(const Minmod& lhs, const Minmod& rhs) noexcept;

}  // namespace grmhd::ValenciaDivClean::Limiters
