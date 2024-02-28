// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrSchildCoords.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma:  no_include <pup.h>

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {

/*!
 * \brief Magnetized fluid disk orbiting a Kerr black hole.
 *
 * In the context of simulating accretion disks, this class implements a widely
 * used (e.g. \cite Gammie2003, \cite Porth2016rfi, \cite White2015omx)
 * initial setup for the GRMHD variables, consisting of a Fishbone-Moncrief disk
 * \cite Fishbone1976apj (see also
 * RelativisticEuler::Solutions::FishboneMoncriefDisk),
 * threaded by a weak poloidal magnetic field. The magnetic field is constructed
 * from an axially symmetric toroidal magnetic potential which, in Kerr
 * ("spherical Kerr-Schild") coordinates, has the form
 *
 * \f{align*}
 * A_\phi(r,\theta) \propto \text{max}(\rho(r,\theta) - \rho_\text{thresh}, 0),
 * \f}
 *
 * where \f$\rho_\text{thresh}\f$ is a user-specified threshold density that
 * confines the magnetic flux to exist inside of the fluid disk only. A commonly
 * used value for this parameter is
 * \f$\rho_\text{thresh} = 0.2\rho_\text{max}\f$, where \f$\rho_\text{max}\f$
 * is the maximum value of
 * the rest mass density in the disk. Using this potential, the Eulerian
 * magnetic field takes the form
 *
 * \f{align*}
 * B^r = \frac{F_{\theta\phi}}{\sqrt{\gamma}},\quad
 * B^\theta = \frac{F_{\phi r}}{\sqrt{\gamma}},\quad B^\phi = 0,
 * \f}
 *
 * where \f$F_{ij} = \partial_i A_j - \partial_j A_i\f$ are the spatial
 * components of the Faraday tensor, and \f$\gamma\f$ is the determinant of the
 * spatial metric. The magnetic field is then normalized so that the
 * plasma-\f$\beta\f$ parameter, \f$\beta = 2p/b^2\f$, equals some value
 * specified by the user. Here, \f$p\f$ is the fluid pressure, and
 *
 * \f{align*}
 * b^2 = b^\mu b_\mu = \frac{B_iB^i}{W^2} + (B^iv_i)^2
 * \f}
 *
 * is the norm of the magnetic field in the fluid frame, with \f$v_i\f$ being
 * the spatial velocity, and \f$W\f$ the Lorentz factor.
 *
 * \note When using Kerr-Schild coordinates, the horizon that is at
 * constant \f$r\f$ is not spherical, but instead spheroidal. This could make
 * application of boundary condition and computing various fluxes
 * across the horizon more complicated than they need to be.
 * Thus, similar to RelativisticEuler::Solutions::FishboneMoncriefDisk
 * we use Spherical Kerr-Schild coordinates,
 * see gr::Solutions::SphericalKerrSchild, in which constant \f$r\f$
 * is spherical. Because we compute variables in Kerr-Schild coordinates,
 * there is a necessary extra step of transforming them back to
 * Spherical Kerr-Schild coordinates.
 *
 * \warning Spherical Kerr-Schild coordinates and "spherical Kerr-Schild"
 * coordinates are not same.
 *
 */
class MagnetizedFmDisk : public virtual evolution::initial_data::InitialData,
                         public MarkAsAnalyticData {
 private:
  using FmDisk = RelativisticEuler::Solutions::FishboneMoncriefDisk;

 public:
  /// The rest mass density (in units of the maximum rest mass density in the
  /// disk) below which the matter in the disk is initially unmagetized.
  struct ThresholdDensity {
    using type = double;
    static constexpr Options::String help = {
        "Frac. rest mass density below which B-field vanishes."};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };
  /// The maximum-magnetic-pressure-to-maximum-fluid-pressure ratio.
  struct InversePlasmaBeta {
    using type = double;
    static constexpr Options::String help = {
        "Ratio of max magnetic pressure to max fluid pressure."};
    static type lower_bound() { return 0.0; }
  };
  /// Grid resolution used in magnetic field normalization.
  struct BFieldNormGridRes {
    using type = size_t;
    static constexpr Options::String help = {
        "Grid Resolution for b-field normalization."};
    static type suggested_value() { return 255; }
    static type lower_bound() { return 4; }
  };

  // Unlike the other analytic data classes, we cannot get these from the
  // `AnalyticDataBase` because this case causes clang-tidy to believe that
  // there is an ambiguous inheritance problem
  static constexpr size_t volume_dim = 3_st;

  template <typename DataType>
  using tags =
      tmpl::push_back<typename gr::AnalyticSolution<3>::template tags<DataType>,
                      hydro::Tags::RestMassDensity<DataType>,
                      hydro::Tags::SpecificInternalEnergy<DataType>,
                      hydro::Tags::Temperature<DataType>,
                      hydro::Tags::Pressure<DataType>,
                      hydro::Tags::SpatialVelocity<DataType, 3>,
                      hydro::Tags::MagneticField<DataType, 3>,
                      hydro::Tags::DivergenceCleaningField<DataType>,
                      hydro::Tags::LorentzFactor<DataType>,
                      hydro::Tags::SpecificEnthalpy<DataType>>;

  using options = tmpl::push_back<FmDisk::options, ThresholdDensity,
                                  InversePlasmaBeta, BFieldNormGridRes>;

  static constexpr Options::String help = {
      "Magnetized Fishbone-Moncrief disk."};

  MagnetizedFmDisk() = default;
  MagnetizedFmDisk(const MagnetizedFmDisk& /*rhs*/) = default;
  MagnetizedFmDisk& operator=(const MagnetizedFmDisk& /*rhs*/) = default;
  MagnetizedFmDisk(MagnetizedFmDisk&& /*rhs*/) = default;
  MagnetizedFmDisk& operator=(MagnetizedFmDisk&& /*rhs*/) = default;
  ~MagnetizedFmDisk() override = default;

  MagnetizedFmDisk(
      double bh_mass, double bh_dimless_spin, double inner_edge_radius,
      double max_pressure_radius, double polytropic_constant,
      double polytropic_exponent, double threshold_density,
      double inverse_plasma_beta,
      size_t normalization_grid_res = BFieldNormGridRes::suggested_value());

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit MagnetizedFmDisk(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(MagnetizedFmDisk);
  /// \endcond

  // Overload the variables function from the base class.
  using equation_of_state_type = typename FmDisk::equation_of_state_type;

  /// @{
  /// The variables in Cartesian Kerr-Schild coordinates at `(x, t)`.
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    // Can't store IntermediateVariables as member variable because we
    // need to be threadsafe.
    typename FmDisk::IntermediateVariables<DataType> vars(x);
    return {std::move(get<Tags>([this, &x, &vars]() {
      if constexpr (std::is_same_v<hydro::Tags::MagneticField<DataType, 3>,
                                   Tags>) {
        return variables(x, tmpl::list<Tags>{}, make_not_null(&vars));
      } else {
        return fm_disk_.variables(x, tmpl::list<Tags>{}, make_not_null(&vars));
      }
    }()))...};
  }

  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    // Can't store IntermediateVariables as member variable because we need to
    // be threadsafe.
    typename FmDisk::IntermediateVariables<DataType> vars(x);
    if constexpr (std::is_same_v<hydro::Tags::MagneticField<DataType, 3>,
                                 Tag>) {
      return variables(x, tmpl::list<Tag>{}, make_not_null(&vars));
    } else {
      return fm_disk_.variables(x, tmpl::list<Tag>{}, make_not_null(&vars));
    }
  }
  /// @}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override;

  const equation_of_state_type& equation_of_state() const {
    return fm_disk_.equation_of_state();
  }

 private:
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/,
                 gsl::not_null<FmDisk::IntermediateVariables<DataType>*> vars)
      const -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  tnsr::I<DataType, 3> unnormalized_magnetic_field(
      const tnsr::I<DataType, 3>& x) const;

  friend bool operator==(const MagnetizedFmDisk& lhs,
                         const MagnetizedFmDisk& rhs);

  RelativisticEuler::Solutions::FishboneMoncriefDisk fm_disk_{};
  double threshold_density_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_plasma_beta_ = std::numeric_limits<double>::signaling_NaN();
  double b_field_normalization_ = std::numeric_limits<double>::signaling_NaN();
  size_t normalization_grid_res_ = 255;
  gr::KerrSchildCoords kerr_schild_coords_{};
};

bool operator!=(const MagnetizedFmDisk& lhs, const MagnetizedFmDisk& rhs);

}  // namespace grmhd::AnalyticData
