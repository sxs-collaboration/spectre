// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
/// \endcond

namespace RadiationTransport::M1Grey::AnalyticData {
/*!
 * \brief Construct a homogeneous sphere of neutrino radiation.
 *
 * We follow the homogeneous sphere test problem in Section 4.5 of
 * \cite radice2022.  The initial data has radius = 1, with equal emissivity
 * and absorption \f$\eta = \kappa_a = 10\f$ inside the uniform sphere.
 * Outside of the sphere the absorption is much lower, allowing the neutrinos
 * to stream out.  Initially the neutrino energy density is distributed
 * uniformly inside the sphere. The momentum density is initialized to 0.
 *
 * Note:
 * To avoid sharp discontinuities, we round the edges of the energy profile
 * with an arctangent function, instead of the step function, which has
 * sharper features.
 */
class HomogeneousSphere : public virtual evolution::initial_data::InitialData,
                          public MarkAsAnalyticData {
 public:
  static constexpr Options::String help = {
      "A homogeneous sphere emitting and absorbing neutrinos."};

  /// The sphere radius.
  struct Radius {
    using type = double;
    static constexpr Options::String help = "Sphere radius";
  };

  /// The emissivity and absorption opacity.
  struct EmissivityAndOpacity {
    using type = double;
    static constexpr Options::String help = "Emissivity and absorption opacity";
  };

  /// The absorption opacity of the exterior
  struct OuterOpacity {
    using type = double;
    static constexpr Options::String help =
        "Opacity of outer absorption region";
  };

  /// BoundaryRoundness
  struct BoundaryRoundness {
    using type = double;
    static constexpr Options::String help =
        "How rounded the interface is between the sphere radius and outer "
        "region.  Closer to 0 corresponds to sharper (more step like) "
        "interface.";
    static type lower_bound() { return 1e-5; }
  };

  using options =
      tmpl::list<Radius, EmissivityAndOpacity, OuterOpacity, BoundaryRoundness>;

  HomogeneousSphere() = default;

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(HomogeneousSphere);
  /// \endcond

  HomogeneousSphere(double radius, double emissivity_and_opacity,
                    double outer_opacity, double boundary_roundness);

  /// @{
  /// Retrieve fluid and neutrino variables
  template <typename NeutrinoSpecies>
  auto variables(const tnsr::I<DataVector, 3>& x,
                 tmpl::list<RadiationTransport::M1Grey::Tags::TildeE<
                     Frame::Inertial, NeutrinoSpecies>> /*meta*/) const
      -> tuples::TaggedTuple<RadiationTransport::M1Grey::Tags::TildeE<
          Frame::Inertial, NeutrinoSpecies>>;

  template <typename NeutrinoSpecies>
  auto variables(const tnsr::I<DataVector, 3>& x,
                 tmpl::list<RadiationTransport::M1Grey::Tags::TildeS<
                     Frame::Inertial, NeutrinoSpecies>> /*meta*/) const
      -> tuples::TaggedTuple<RadiationTransport::M1Grey::Tags::TildeS<
          Frame::Inertial, NeutrinoSpecies>>;

  template <typename NeutrinoSpecies>
  auto variables(const tnsr::I<DataVector, 3>& x,
                 tmpl::list<RadiationTransport::M1Grey::Tags::GreyEmissivity<
                     NeutrinoSpecies>> /*meta*/) const
      -> tuples::TaggedTuple<
          RadiationTransport::M1Grey::Tags::GreyEmissivity<NeutrinoSpecies>>;

  template <typename NeutrinoSpecies>
  auto variables(
      const tnsr::I<DataVector, 3>& x,
      tmpl::list<RadiationTransport::M1Grey::Tags::GreyAbsorptionOpacity<
          NeutrinoSpecies>> /*meta*/) const
      -> tuples::TaggedTuple<RadiationTransport::M1Grey::Tags::
                                 GreyAbsorptionOpacity<NeutrinoSpecies>>;

  template <typename NeutrinoSpecies>
  auto variables(
      const tnsr::I<DataVector, 3>& x,
      tmpl::list<RadiationTransport::M1Grey::Tags::GreyScatteringOpacity<
          NeutrinoSpecies>> /*meta*/) const
      -> tuples::TaggedTuple<RadiationTransport::M1Grey::Tags::
                                 GreyScatteringOpacity<NeutrinoSpecies>>;

  static auto variables(
      const tnsr::I<DataVector, 3>& x,
      tmpl::list<hydro::Tags::LorentzFactor<DataVector>> /*meta*/)
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataVector>>;

  static auto variables(
      const tnsr::I<DataVector, 3>& x,
      tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>> /*meta*/)
      -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataVector, 3>>;
  /// @}

  /// Retrieve the metric variables
  template <typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataVector, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    return gr::Solutions::Minkowski<3>{}.variables(x, 0.0, tmpl::list<Tag>{});
  }

  /// Retrieve a collection of variables
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    return {get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const HomogeneousSphere& lhs,
                         const HomogeneousSphere& rhs) {
    return lhs.radius_ == rhs.radius_ and
           lhs.emissivity_and_opacity_ == rhs.emissivity_and_opacity_ and
           lhs.outer_opacity_ == rhs.outer_opacity_ and
           lhs.boundary_roundness_ == rhs.boundary_roundness_;
  }

  double radius_ = std::numeric_limits<double>::signaling_NaN();
  double emissivity_and_opacity_ = std::numeric_limits<double>::signaling_NaN();
  double outer_opacity_ = std::numeric_limits<double>::signaling_NaN();
  double boundary_roundness_ = std::numeric_limits<double>::signaling_NaN();
};

bool operator!=(const HomogeneousSphere& lhs, const HomogeneousSphere& rhs);

}  // namespace RadiationTransport::M1Grey::AnalyticData
