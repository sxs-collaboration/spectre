// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {
namespace magnetized_tov_detail {

using StarRegion = RelativisticEuler::Solutions::tov_detail::StarRegion;

template <typename DataType, StarRegion Region>
struct MagnetizedTovVariables
    : RelativisticEuler::Solutions::tov_detail::TovVariables<DataType, Region> {
  static constexpr size_t Dim = 3;
  using Base =
      RelativisticEuler::Solutions::tov_detail::TovVariables<DataType, Region>;
  using Cache = typename Base::Cache;
  using Base::operator();
  using Base::coords;
  using Base::eos;
  using Base::radial_solution;
  using Base::radius;

  size_t pressure_exponent;
  double cutoff_pressure;
  double vector_potential_amplitude;

  MagnetizedTovVariables(
      const tnsr::I<DataType, 3>& local_x, const DataType& local_radius,
      const gr::Solutions::TovSolution& local_radial_solution,
      const EquationsOfState::PolytropicFluid<true>& local_eos,
      size_t local_pressure_exponent, double local_cutoff_pressure,
      double local_vector_potential_amplitude)
      : Base(local_x, local_radius, local_radial_solution, local_eos),
        pressure_exponent(local_pressure_exponent),
        cutoff_pressure(local_cutoff_pressure),
        vector_potential_amplitude(local_vector_potential_amplitude) {}

  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
      gsl::not_null<Cache*> cache,
      hydro::Tags::MagneticField<DataType, 3> /*meta*/) const override;
};

}  // namespace magnetized_tov_detail

/*!
 * \brief Magnetized TOV star initial data, where metric terms only account for
 * the hydrodynamics not the magnetic fields.
 *
 * Superposes a poloidal magnetic field on top of a TOV solution where the
 * vector potential has the form
 *
 * \f{align*}{
 *  A_{\phi} = A_b \varpi^2 \max(p-p_{\mathrm{cut}}, 0)^{n_s}
 * \f}
 *
 * where \f$A_b\f$ controls the amplitude of the magnetic field,
 * \f$\varpi^2=x^2+y^2=r^2-z^2\f$ is the cylindrical radius,
 * \f$n_s\f$ controls the degree of differentiability, and
 * \f$p_{\mathrm{cut}}\f$ controls the pressure cutoff below which the magnetic
 * field is zero.
 *
 * In Cartesian coordinates the vector potential is:
 *
 * \f{align*}{
 *   A_x&=-\frac{y}{\varpi^2}A_\phi, \\
 *   A_y&=\frac{x}{\varpi^2}A_\phi, \\
 *   A_z&=0,
 * \f}
 *
 * For the poloidal field this means
 *
 * \f{align*}{
 *   A_x &= -y A_b\max(p-p_{\mathrm{cut}}, 0)^{n_s}, \\
 *   A_y &= x A_b\max(p-p_{\mathrm{cut}}, 0)^{n_s}.
 * \f}
 *
 * The magnetic field is computed from the vector potential as
 *
 * \f{align*}{
 *   B^i&=n_a\epsilon^{aijk}\partial_jA_k \\
 *      &=-\frac{1}{\sqrt{\gamma}}[ijk]\partial_j A_k,
 * \f}
 *
 * where \f$[ijk]\f$ is the total anti-symmetric symbol. This means that
 *
 * \f{align*}{
 *   B^x&=\frac{1}{\sqrt{\gamma}} (\partial_z A_y-\partial_y A_z), \\
 *   B^y&=\frac{1}{\sqrt{\gamma}} (\partial_x A_z - \partial_z A_x), \\
 *   B^z&=\frac{1}{\sqrt{\gamma}} (\partial_y A_x - \partial_x A_y).
 * \f}
 *
 * Focusing on the region where the field is non-zero we have:
 *
 * \f{align*}{
 *   \partial_x A_y
 *   &= A_b(p-p_{\mathrm{cut}})^{n_s}+
 *     \frac{x^2}{r}A_bn_s(p-p_{\mathrm{cut}})^{n_s-1}\partial_r p \\
 *   \partial_y A_x
 *   &= -A_b(p-p_{\mathrm{cut}})^{n_s}-
 *     \frac{y^2}{r}A_bn_s(p-p_{\mathrm{cut}})^{n_s-1}\partial_r p \\
 *   \partial_z A_x
 *   &= -\frac{yz}{r}A_bn_s(p-p_{\mathrm{cut}})^{n_s-1}\partial_rp \\
 *   \partial_z A_y
 *   &= \frac{xz}{r}A_bn_s(p-p_{\mathrm{cut}})^{n_s-1}\partial_rp
 * \f}
 *
 * The magnetic field is given by:
 *
 * \f{align*}{
 *   B^x&=\frac{1}{\sqrt{\gamma}}\frac{xz}{r}
 *        A_bn_s(p-p_{\mathrm{cut}})^{n_s-1}\partial_rp \\
 *   B^y&=\frac{1}{\sqrt{\gamma}}\frac{yz}{r}
 *        A_bn_s(p-p_{\mathrm{cut}})^{n_s-1}\partial_rp \\
 *   B^z&=-\frac{A_b}{\sqrt{\gamma}}\left[
 *        2(p-p_{\mathrm{cut}})^{n_s} \phantom{\frac{a}{b}}\right. \\
 *      &\left.+\frac{x^2+y^2}{r}
 *        n_s(p-p_{\mathrm{cut}})^{n_s-1}\partial_r p
 *        \right]
 * \f}
 *
 * Taking the small-\f$r\f$ limit gives the \f$r=0\f$ magnetic field:
 *
 * \f{align*}{
 *   B^x&=0, \\
 *   B^y&=0, \\
 *   B^z&=-\frac{A_b}{\sqrt{\gamma}}
 *        2(p-p_{\mathrm{cut}})^{n_s}.
 * \f}
 *
 * While the amplitude \f$A_b\f$ is specified in the code, it is more natural
 * to work with the magnetic field strength, which is given by \f$\sqrt{b^2}\f$
 * (where \f$b^a\f$ is the comoving magnetic field), and in CGS units is
 *
 * \f{align*}{
 *  |B_{\mathrm{CGS}}|&= \sqrt{4 \pi b^2}
 *   \left(\frac{c^2}{G M_\odot}\right) \left(\frac{c}{\sqrt{4 \pi \epsilon_0
 *    G}}\right) \\
 *   &= \sqrt{b^2} \times 8.352\times10^{19}\mathrm{G} \,.
 * \f}
 *
 * We now give values used for standard tests of magnetized stars.
 * - \f$\rho_c(0)=1.28\times10^{-3}\f$
 * - \f$K=100\f$
 * - \f$\Gamma=2\f$
 * - %Domain \f$[-20,20]^3\f$
 * - Units \f$M=M_\odot\f$
 * - A target final time 20ms means \f$20\times10^{-3}/(5\times10^{-6})=4000M\f$
 * - The mass of the star is \f$1.4M_{\odot}\f$
 *
 * Parameters for desired magnetic field strength:
 * - For \f$n_s=0\f$ and \f$p_{\mathrm{cut}}=0.04p_{\max}\f$ setting
 *   \f$A_b=6\times10^{-5}\f$ yields a maximum mganetic field strength of
 *   \f$1.002\times10^{16}G\f$.
 * - For \f$n_s=1\f$ and \f$p_{\mathrm{cut}}=0.04p_{\max}\f$ setting
 *   \f$A_b=0.4\f$ yields a maximum mganetic field strength of
 *   \f$1.05\times10^{16}G\f$.
 * - For \f$n_s=2\f$ and \f$p_{\mathrm{cut}}=0.04p_{\max}\f$ setting
 *   \f$A_b=2500\f$ yields a maximum mganetic field strength of
 *   \f$1.03\times10^{16}G\f$.
 * - For \f$n_s=3\f$ and \f$p_{\mathrm{cut}}=0.04p_{\max}\f$ setting
 *   \f$A_b=1.65\times10^{7}\f$ yields a maximum mganetic field strength of
 *   \f$1.07\times10^{16}G\f$.
 *
 * Note that the magnetic field strength goes as \f$A_b\f$ so any desired value
 * can be achieved by a linear scaling.
 */
class MagnetizedTovStar : public virtual evolution::initial_data::InitialData,
                          public MarkAsAnalyticData,
                          private RelativisticEuler::Solutions::TovStar {
 private:
  using tov_star = RelativisticEuler::Solutions::TovStar;

 public:
  struct PressureExponent {
    using type = size_t;
    static constexpr Options::String help = {
        "The exponent n_s controlling the smoothness of the field"};
  };

  struct VectorPotentialAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "The amplitude A_b of the phi-component of the vector potential. This "
        "controls the magnetic field strength."};
    static type lower_bound() { return 0.0; }
  };

  struct CutoffPressureFraction {
    using type = double;
    static constexpr Options::String help = {
        "The fraction of the central pressure below which there is no magnetic "
        "field. p_cut = Fraction * p_max."};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };

  using options =
      tmpl::push_back<tov_star::options, PressureExponent,
                      CutoffPressureFraction, VectorPotentialAmplitude>;

  static constexpr Options::String help = {"Magnetized TOV star."};

  static constexpr size_t volume_dim = 3_st;

  template <typename DataType>
  using tags = typename tov_star::template tags<DataType>;

  MagnetizedTovStar() = default;
  MagnetizedTovStar(const MagnetizedTovStar& /*rhs*/) = default;
  MagnetizedTovStar& operator=(const MagnetizedTovStar& /*rhs*/) = default;
  MagnetizedTovStar(MagnetizedTovStar&& /*rhs*/) = default;
  MagnetizedTovStar& operator=(MagnetizedTovStar&& /*rhs*/) = default;
  ~MagnetizedTovStar() = default;

  MagnetizedTovStar(double central_rest_mass_density,
                    double polytropic_constant, double polytropic_exponent,
                    gr::Solutions::TovCoordinates coordinate_system,
                    size_t pressure_exponent, double cutoff_pressure_fraction,
                    double vector_potential_amplitude);

  /// \cond
  explicit MagnetizedTovStar(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(MagnetizedTovStar);
  /// \endcond

  using tov_star::equation_of_state;
  using tov_star::equation_of_state_type;

  /// Retrieve a collection of variables at `(x)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    return variables_impl<magnetized_tov_detail::MagnetizedTovVariables>(
        x, tmpl::list<Tags...>{}, pressure_exponent_, cutoff_pressure_,
        vector_potential_amplitude_);
  }

  void pup(PUP::er& p);  //  NOLINT

 private:
  friend bool operator==(const MagnetizedTovStar& lhs,
                         const MagnetizedTovStar& rhs);

 protected:
  size_t pressure_exponent_ = std::numeric_limits<size_t>::max();
  double cutoff_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double vector_potential_amplitude_ =
      std::numeric_limits<double>::signaling_NaN();
};

bool operator!=(const MagnetizedTovStar& lhs, const MagnetizedTovStar& rhs);
}  // namespace grmhd::AnalyticData
