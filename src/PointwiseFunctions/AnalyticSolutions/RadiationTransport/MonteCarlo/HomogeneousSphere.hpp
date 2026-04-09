// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <memory>
#include <pup.h>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace RadiationTransport::MonteCarlo::Solutions {

/*!
 * \brief Homogeneous sphere as fluid background to MC run
 *
 * Provides background fluid variables for a
 * fluid with constant density, temperature, Ye
 * in Minkowski spacetime.
 *
 */
class HomogeneousSphere : public evolution::initial_data::InitialData,
                          public MarkAsAnalyticSolution {
 public:
  constexpr static bool IsRelativistic = true;
  using equation_of_state_type =
      EquationsOfState::EquationOfState<IsRelativistic, 3>;

  const EquationsOfState::EquationOfState<IsRelativistic, 3>&
  equation_of_state() const {
    return *equation_of_state_;
  }

  static const size_t volume_dim = 3;

  /// The radius of the sphere
  struct Radius {
    using type = double;
    static constexpr Options::String help = {"The radius of the sphere."};
  };
  /// The density inside and outside the sphere
  struct Densities {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {"Density inside and outside."};
  };
  /// The temperature inside and outside the sphere
  struct Temperatures {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {"Temperature inside and outside."};
  };
  /// The electron fraction inside and outside the sphere
  struct ElectronFractions {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {"Ye inside and outside."};
  };

  using options = tmpl::list<
      Radius, Densities, Temperatures, ElectronFractions,
      hydro::OptionTags::InitialDataEquationOfState<IsRelativistic, 3>>;
  static constexpr Options::String help = {
      "Background for uniform sphere with constant rho, T, Ye"};

  HomogeneousSphere() = default;
  HomogeneousSphere(const HomogeneousSphere& /*rhs*/);
  HomogeneousSphere& operator=(const HomogeneousSphere& /*rhs*/);
  HomogeneousSphere(HomogeneousSphere&& /*rhs*/) = default;
  HomogeneousSphere& operator=(HomogeneousSphere&& /*rhs*/) = default;
  ~HomogeneousSphere() override = default;

  HomogeneousSphere(
      const double& radius, const std::array<double, 2>& densities,
      const std::array<double, 2>& temperatures,
      const std::array<double, 2>& electron_fractions,
      std::unique_ptr<EquationsOfState::EquationOfState<IsRelativistic, 3>>
          local_eos);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit HomogeneousSphere(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(HomogeneousSphere);
  /// \endcond

  /// @{
  /// Retrieve fluid variables at `(x, t)`
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double /*t*/,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double /*t*/,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;
  /// @}

  /// Retrieve a collection of hydro variables at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         double t,
                                         tmpl::list<Tags...> /*meta*/) const {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag,
            Requires<not tmpl::list_contains_v<
                tmpl::push_back<hydro::grmhd_tags<DataType>,
                                hydro::Tags::SpecificEnthalpy<DataType>>,
                Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     double t, tmpl::list<Tag> /*meta*/) const {
    return background_spacetime_.variables(x, t, tmpl::list<Tag>{});
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override;

 private:
  friend bool operator==(const HomogeneousSphere& lhs,
                         const HomogeneousSphere& rhs);

  double radius_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 2> densities_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  std::array<double, 2> temperatures_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  std::array<double, 2> electron_fractions_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};

  std::unique_ptr<equation_of_state_type> equation_of_state_;
  gh::Solutions::WrappedGr<gr::Solutions::Minkowski<3>>
      background_spacetime_{};
};

bool operator!=(const HomogeneousSphere& lhs, const HomogeneousSphere& rhs);
}  // namespace RadiationTransport::MonteCarlo::Solutions
