// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrSchildCoords.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma:  no_include <pup.h>

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace grmhd {
namespace AnalyticData {

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
 */
class MagnetizedFmDisk
    : public RelativisticEuler::Solutions::FishboneMoncriefDisk {
 private:
  using fm_disk = RelativisticEuler::Solutions::FishboneMoncriefDisk;

 public:
  /// The rest mass density (in units of the maximum rest mass density in the
  /// disk) below which the matter in the disk is initially unmagetized.
  struct ThresholdDensity {
    using type = double;
    static constexpr OptionString help = {
        "Frac. rest mass density below which B-field vanishes."};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };
  /// The maximum-magnetic-pressure-to-maximum-fluid-pressure ratio.
  struct InversePlasmaBeta {
    using type = double;
    static constexpr OptionString help = {
        "Ratio of max magnetic pressure to max fluid pressure."};
    static type lower_bound() { return 0.0; }
  };
  /// Grid resolution used in magnetic field normalization.
  struct BFieldNormGridRes {
    using type = size_t;
    static constexpr OptionString help = {
        "Grid Resolution for b-field normalization."};
    static type default_value() { return 255; }
    static type lower_bound() { return 4; }
  };

  using options = tmpl::push_back<fm_disk::options, ThresholdDensity,
                                  InversePlasmaBeta, BFieldNormGridRes>;

  static constexpr OptionString help = {"Magnetized Fishbone-Moncrief disk."};

  MagnetizedFmDisk() = default;
  MagnetizedFmDisk(const MagnetizedFmDisk& /*rhs*/) = delete;
  MagnetizedFmDisk& operator=(const MagnetizedFmDisk& /*rhs*/) = delete;
  MagnetizedFmDisk(MagnetizedFmDisk&& /*rhs*/) noexcept = default;
  MagnetizedFmDisk& operator=(MagnetizedFmDisk&& /*rhs*/) noexcept = default;
  ~MagnetizedFmDisk() = default;

  MagnetizedFmDisk(double bh_mass, double bh_dimless_spin,
                   double inner_edge_radius, double max_pressure_radius,
                   double polytropic_constant, double polytropic_exponent,
                   double threshold_density, double inverse_plasma_beta,
                   size_t normalization_grid_res =
                       BFieldNormGridRes::default_value()) noexcept;

  // Overload the variables function from the base class.
  using fm_disk::variables;

  // @{
  /// The grmhd variables in Cartesian Kerr-Schild coordinates at `(x, t)`
  ///
  /// \note The functions are optimized for retrieving the hydro variables
  /// before the metric variables.
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    // Can't store IntermediateVariables as member variable because we
    // need to be threadsafe.
    constexpr double dummy_time = 0.0;
    IntermediateVariables<
        DataType,
        tmpl2::flat_any_v<(
            cpp17::is_same_v<Tags, hydro::Tags::SpatialVelocity<DataType, 3>> or
            cpp17::is_same_v<Tags, hydro::Tags::LorentzFactor<DataType>> or
            not tmpl::list_contains_v<hydro::grmhd_tags<DataType>, Tags>)...>>
        vars(bh_spin_a_, background_spacetime_, x, dummy_time,
             index_helper(
                 tmpl::index_of<tmpl::list<Tags...>,
                                hydro::Tags::SpatialVelocity<DataType, 3>>{}),
             index_helper(
                 tmpl::index_of<tmpl::list<Tags...>,
                                hydro::Tags::LorentzFactor<DataType>>{}));
    return {std::move(get<Tags>(
        variables(x, tmpl::list<Tags>{}, vars,
                  tmpl::index_of<tmpl::list<Tags...>, Tags>::value)))...};
  }

  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    // Can't store IntermediateVariables as member variable because we need to
    // be threadsafe.
    constexpr double dummy_time = 0.0;
    IntermediateVariables<
        DataType,
        cpp17::is_same_v<Tag, hydro::Tags::SpatialVelocity<DataType, 3>> or
            cpp17::is_same_v<Tag, hydro::Tags::LorentzFactor<DataType>> or
            not tmpl::list_contains_v<hydro::grmhd_tags<DataType>, Tag>>
        intermediate_vars(bh_spin_a_, background_spacetime_, x, dummy_time,
                          std::numeric_limits<size_t>::max(),
                          std::numeric_limits<size_t>::max());
    return variables(x, tmpl::list<Tag>{}, intermediate_vars, 0);
  }
  // @}

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

 private:
  template <typename DataType, bool NeedSpacetime>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>> /*meta*/,
      const IntermediateVariables<DataType, NeedSpacetime>& vars,
      size_t index) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  tnsr::I<DataType, 3, Frame::Inertial> unnormalized_magnetic_field(
      const tnsr::I<DataType, 3, Frame::Inertial>& x) const noexcept;

  friend bool operator==(const MagnetizedFmDisk& lhs,
                         const MagnetizedFmDisk& rhs) noexcept;

  double threshold_density_ = std::numeric_limits<double>::signaling_NaN();
  double inverse_plasma_beta_ = std::numeric_limits<double>::signaling_NaN();
  double b_field_normalization_ = std::numeric_limits<double>::signaling_NaN();
  size_t normalization_grid_res_ = 255;
  gr::KerrSchildCoords kerr_schild_coords_{};
};

bool operator!=(const MagnetizedFmDisk& lhs,
                const MagnetizedFmDisk& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace grmhd
