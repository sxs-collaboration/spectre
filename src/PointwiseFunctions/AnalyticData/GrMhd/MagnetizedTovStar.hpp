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
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/Poloidal.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/Toroidal.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
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

  size_t poloidal_pressure_exponent;
  double poloidal_cutoff_pressure;
  double poloidal_vector_potential_amplitude;

  size_t toroidal_pressure_exponent;
  double toroidal_cutoff_pressure;
  double toroidal_vector_potential_amplitude;

  MagnetizedTovVariables(
      const tnsr::I<DataType, 3>& local_x, const DataType& local_radius,
      const RelativisticEuler::Solutions::TovSolution& local_radial_solution,
      const EquationsOfState::EquationOfState<true, 1>& local_eos,
      size_t local_poloidal_pressure_exponent,
      double local_poloidal_cutoff_pressure,
      double local_poloidal_vector_potential_amplitude,
      size_t local_toroidal_pressure_exponent,
      double local_toroidal_cutoff_pressure,
      double local_toroidal_vector_potential_amplitude)
      : Base(local_x, local_radius, local_radial_solution, local_eos),
        poloidal_pressure_exponent(local_poloidal_pressure_exponent),
        poloidal_cutoff_pressure(local_poloidal_cutoff_pressure),
        poloidal_vector_potential_amplitude(
            local_poloidal_vector_potential_amplitude),
        toroidal_pressure_exponent(local_toroidal_pressure_exponent),
        toroidal_cutoff_pressure(local_toroidal_cutoff_pressure),
        toroidal_vector_potential_amplitude(
            local_toroidal_vector_potential_amplitude) {}

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
 * Superposes poloidal and toroidal magnetic fields on top of a TOV solution.
 * For details see the documentation of `Poloidal` and `Toroidal` initial
 * magnetic fields.
 *
 * While the amplitudes \f$A_b\f$ are specified in the code, it is more natural
 * to work with the magnetic field strength, which is given by \f$\sqrt{b^2}\f$
 * (where \f$b^a\f$ is the comoving magnetic field), and in CGS units is
 *
 * \f{align*}{
 *  |B_{\mathrm{CGS}}|
 *      &= \sqrt{b^2} \sqrt{4\pi} \left(\frac{c^4}{G^{3/2} M_\odot}\right) \\
 *      &= \sqrt{b^2} \times 8.35\times10^{19} \, \mathrm{Gauss} \,.
 * \f}
 *
 * Note the extra factor \f$\sqrt{4\pi}\f$, since equations in ValenciaDivClean
 * system are using the geometrised Heaviside-Lorentz unit convention in which
 * the factor \f$4\pi\f$ does not appear in the Maxwell's equations.
 *
 * We now give example values for desired magnetic field strengths for the
 * following TOV star:
 *
 * - \f$\rho_c(0)=1.28\times10^{-3}\f$
 * - \f$K=100\f$
 * - \f$\Gamma=2\f$
 * - The mass of the star is about \f$1.4M_{\odot}\f$
 *
 * For purely poloidal field with \f$p_{\mathrm{cut}}=0.03 p_{\max}\f$,
 *
 * <table>
 * <caption> Poloidal field </caption>
 * <tr><th> \f$n_s\f$ <th> \f$A_b\f$ <th> Max field strength (Gauss)
 * <tr><td> 0 <td> \f$5 \times 10^{-5}\f$ <td> \f$5.55 \times 10^{15}\f$
 * <tr><td> 1 <td> \f$0.4\f$ <td> \f$5.22 \times 10^{15}\f$
 * <tr><td> 2 <td> \f$2500\f$ <td> \f$5.19 \times 10^{15}\f$
 * <tr><td> 3 <td> \f$1.65 \times 10^{7}\f$ <td> \f$5.44 \times 10^{15}\f$
 * </table>
 *
 * For purely toroidal field with \f$p_{\mathrm{cut}}=0.03 p_{\max}\f$,
 *
 * <table>
 * <caption> Toroidal field </caption>
 * <tr><th> \f$n_s\f$ <th> \f$A_b\f$ <th> Max field strength (Gauss)
 * <tr><td> 0 <td> \f$1 \times 10^{-5}\f$ <td> \f$7.14 \times 10^{15}\f$
 * <tr><td> 1 <td> \f$0.1\f$ <td> \f$1.72 \times 10^{15}\f$
 * <tr><td> 2 <td> \f$1000\f$ <td> \f$1.52 \times 10^{15}\f$
 * <tr><td> 3 <td> \f$1.0 \times 10^{7}\f$ <td> \f$1.96 \times 10^{15}\f$
 * </table>
 *
 * Note that for purely poloidal or toroidal cases, the magnetic field strength
 * goes as \f$A_b\f$ so any desired value can be achieved by a linear scaling.
 *
 */
class MagnetizedTovStar : public virtual evolution::initial_data::InitialData,
                          public MarkAsAnalyticData,
                          private RelativisticEuler::Solutions::TovStar {
 private:
  using tov_star = RelativisticEuler::Solutions::TovStar;

 public:
  struct PoloidalField {
    static constexpr Options::String help = {
        "Options related to the poloidal component of initial magnetic field."};
  };

  struct PoloidalPressureExponent
      : InitialMagneticFields::Poloidal::PressureExponent {
    using group = PoloidalField;
    static std::string name() { return "PressureExponent"; }
  };

  struct PoloidalCutoffPressureFraction {
    using group = PoloidalField;
    using type = double;
    static std::string name() { return "CutoffPressureFraction"; }
    static constexpr Options::String help = {
        "The fraction of the central pressure below which there is no magnetic "
        "field. p_cut = Fraction * p_max."};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };

  struct PoloidalVectorPotentialAmplitude
      : InitialMagneticFields::Poloidal::VectorPotentialAmplitude {
    using group = PoloidalField;
    static std::string name() { return "VectorPotentialAmplitude"; }
  };

  struct ToroidalField {
    static constexpr Options::String help = {
        "Options related to the toroidal component of initial magnetic field."};
  };

  struct ToroidalPressureExponent
      : InitialMagneticFields::Toroidal::PressureExponent {
    using group = ToroidalField;
    static std::string name() { return "PressureExponent"; }
  };

  struct ToroidalCutoffPressureFraction {
    using group = ToroidalField;
    using type = double;
    static std::string name() { return "CutoffPressureFraction"; }
    static constexpr Options::String help = {
        "The fraction of the central pressure below which there is no magnetic "
        "field. p_cut = Fraction * p_max."};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };

  struct ToroidalVectorPotentialAmplitude
      : InitialMagneticFields::Toroidal::VectorPotentialAmplitude {
    using group = ToroidalField;
    static std::string name() { return "VectorPotentialAmplitude"; }
  };

  using options =
      tmpl::push_back<tov_star::options, PoloidalPressureExponent,
                      PoloidalCutoffPressureFraction,
                      PoloidalVectorPotentialAmplitude,
                      ToroidalPressureExponent, ToroidalCutoffPressureFraction,
                      ToroidalVectorPotentialAmplitude>;

  static constexpr Options::String help = {"Magnetized TOV star."};

  static constexpr size_t volume_dim = 3_st;

  template <typename DataType>
  using tags = typename tov_star::template tags<DataType>;

  MagnetizedTovStar() = default;
  MagnetizedTovStar(const MagnetizedTovStar& /*rhs*/) = default;
  MagnetizedTovStar& operator=(const MagnetizedTovStar& /*rhs*/) = default;
  MagnetizedTovStar(MagnetizedTovStar&& /*rhs*/) = default;
  MagnetizedTovStar& operator=(MagnetizedTovStar&& /*rhs*/) = default;
  ~MagnetizedTovStar() override = default;

  MagnetizedTovStar(
      double central_rest_mass_density,
      std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>
          equation_of_state,
      RelativisticEuler::Solutions::TovCoordinates coordinate_system,
      size_t poloidal_pressure_exponent,
      double poloidal_cutoff_pressure_fraction,
      double poloidal_vector_potential_amplitude,
      size_t toroidal_pressure_exponent,
      double toroidal_cutoff_pressure_fraction,
      double toroidal_vector_potential_amplitude);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

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
        x, tmpl::list<Tags...>{}, poloidal_pressure_exponent_,
        poloidal_cutoff_pressure_, poloidal_vector_potential_amplitude_,
        toroidal_pressure_exponent_, toroidal_cutoff_pressure_,
        toroidal_vector_potential_amplitude_);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const MagnetizedTovStar& lhs,
                         const MagnetizedTovStar& rhs);

 protected:
  size_t poloidal_pressure_exponent_ = std::numeric_limits<size_t>::max();
  double poloidal_cutoff_pressure_ =
      std::numeric_limits<double>::signaling_NaN();
  double poloidal_vector_potential_amplitude_ =
      std::numeric_limits<double>::signaling_NaN();
  size_t toroidal_pressure_exponent_ = std::numeric_limits<size_t>::max();
  double toroidal_cutoff_pressure_ =
      std::numeric_limits<double>::signaling_NaN();
  double toroidal_vector_potential_amplitude_ =
      std::numeric_limits<double>::signaling_NaN();
};

bool operator!=(const MagnetizedTovStar& lhs, const MagnetizedTovStar& rhs);
}  // namespace grmhd::AnalyticData
