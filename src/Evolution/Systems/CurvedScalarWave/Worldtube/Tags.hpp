// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/BackgroundSpacetime.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace OptionTags {
struct InitialTime;
}  // namespace OptionTags
/// \endcond

namespace CurvedScalarWave::Worldtube {
/*!
 * \brief Option tags for the worldtube
 */
namespace OptionTags {
/*!
 * \brief Options for the worldtube
 */
struct Worldtube {
  static constexpr Options::String help = {"Options for the Worldtube"};
};

/*!
 * \brief The value of the scalar charge in units of the black hole mass M.
 */
struct Charge {
  using type = double;
  static constexpr Options::String help{
      "The value of the scalar charge in units of the black hole mass M."};
  using group = Worldtube;
};

/*!
 * \brief Options for the scalar self-force. Select `None` for a purely geodesic
 * evolution
 *
 *\details The self force is turned on using the smooth transition function
 *
 * \begin{equation}
 * w(t) = 1 - \exp{ \left(- \left(\frac{t - t_1}{\sigma} \right)^4 \right)}.
 * \end{equation}
 *
 * The turn on time is given by \f$t_1\f$ and the turn on interval is given by
 *\f$sigma\f$.
 */
struct SelfForceOptions {
  static constexpr Options::String help = {
      "Options for the scalar self-force. Select `None` for a purely geodesic "
      "evolution"};
  using group = Worldtube;
  using type = Options::Auto<SelfForceOptions, Options::AutoLabel::None>;

  struct Mass {
    using type = double;
    static constexpr Options::String help{
        "The mass of the scalar particle in units of the black hole mass M."};
    static double lower_bound() { return 0.; }
  };

  struct Iterations {
    using type = size_t;
    static constexpr Options::String help{
        "The number of iterations used to compute the particle acceleration. "
        "Must be at least 1 as 0 iterations corresponds to the geodesic "
        "acceleration."};
    static size_t lower_bound() { return 1; }
  };

  struct TurnOnTime {
    using type = double;
    static constexpr Options::String help{
        "The time at which the scalar self force is turned on."};
    static double lower_bound() { return 0.; }
  };

  struct TurnOnInterval {
    using type = double;
    static constexpr Options::String help{
        "The interval over which the scalar self force is smoothly turned on. "
        "We require a minimum of 1 M for the interval."};
    static double lower_bound() { return 1.; }
  };

  SelfForceOptions();
  SelfForceOptions(double mass_in, size_t iterations_in, double turn_on_time_in,
                   double turn_on_interval_in);
  void pup(PUP::er& p);

  using options = tmpl::list<Mass, Iterations, TurnOnTime, TurnOnInterval>;

  double mass{};
  size_t iterations{};
  double turn_on_time{};
  double turn_on_interval{};
};

/*!
 * \brief Name of the excision sphere designated to act as a worldtube
 */
struct ExcisionSphere {
  using type = std::string;
  static constexpr Options::String help{
      "The name of the excision sphere as returned by the domain."};
  using group = Worldtube;
};

/*!
 * \brief Triggers at which to write the coefficients of the worldtube's
 * internal Taylor series to file.
 */
struct ObserveCoefficientsTrigger {
  using type = std::unique_ptr<Trigger>;
  static constexpr Options::String help{
      "Specifies a non-dense trigger in which the coefficients of the internal "
      "regular field expansion are written to file."};
  using group = Worldtube;
};

/*!
 * \brief The internal expansion order of the worldtube solution.
 */
struct ExpansionOrder {
  using type = size_t;
  static constexpr Options::String help{
      "The internal expansion order of the worldtube solution. Currently "
      "orders 0 and 1 are implemented"};
  static size_t upper_bound() { return 1; }
  using group = Worldtube;
};
}  // namespace OptionTags

/*!
 * \brief Tags related to the worldtube
 */
namespace Tags {
/*!
 * \brief Dummy tag that throws an error if the input file does not describe a
 * circular orbit.
 */
template <size_t Dim, typename BackgroundType>
struct CheckInputFile : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<
      domain::OptionTags::DomainCreator<Dim>, OptionTags::ExcisionSphere,
      CurvedScalarWave::OptionTags::BackgroundSpacetime<BackgroundType>>;

  // puncture field is specialised on Kerr-Schild bakckground.
  static_assert(std::is_same_v<BackgroundType, gr::Solutions::KerrSchild>);
  static constexpr bool pass_metavariables = false;
  static bool create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator,
      const std::string& excision_sphere_name,
      const BackgroundType& kerr_schild_background) {
    if (not kerr_schild_background.zero_spin()) {
      ERROR(
          "Black hole spin is not implemented yet but you requested non-zero "
          "spin.");
    }
    if (not equal_within_roundoff(kerr_schild_background.center(),
                                  make_array(0., 0., 0.))) {
      ERROR("The central black hole must be centered at [0., 0., 0.].");
    }
    if (not equal_within_roundoff(kerr_schild_background.mass(), 1.)) {
      ERROR("The central black hole must have mass 1.");
    }
    const auto domain = domain_creator->create_domain();
    const auto& excision_spheres = domain.excision_spheres();
    const auto& excision_sphere = excision_spheres.at(excision_sphere_name);
    const double orbital_radius = get<0>(excision_sphere.center());
    const auto& functions_of_time = domain_creator->functions_of_time();
    if (not functions_of_time.count("Rotation")) {
      ERROR("Expected functions of time to contain 'Rotation'.");
    }
    // dynamic cast to access `angle_func_and_deriv` method
    const auto* rotation_function_of_time =
        dynamic_cast<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
            &*functions_of_time.at("Rotation"));
    if (rotation_function_of_time == nullptr) {
      ERROR("Failed dynamic cast to QuaternionFunctionOfTime.");
    }
    const auto angular_velocity =
        rotation_function_of_time->angle_func_and_deriv(0.).at(1);
    if (equal_within_roundoff(orbital_radius, 0.)) {
      ERROR("The orbital radius was set to 0.");
    }
    if (not equal_within_roundoff(
            angular_velocity,
            DataVector{0.0, 0.0, pow(orbital_radius, -1.5)})) {
      ERROR(
          "Only circular orbits are implemented at the moment so the "
          "angular velocity should be [0., 0., orbital_radius^(-3/2)] = "
          << "[0., 0., " << pow(orbital_radius, -1.5) << "]");
    }
    return true;
  }
};

/*!
 * \brief The excision sphere corresponding to the worldtube
 */
template <size_t Dim>
struct ExcisionSphere : db::SimpleTag {
  using type = ::ExcisionSphere<Dim>;
  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>,
                                 OptionTags::ExcisionSphere>;
  static constexpr bool pass_metavariables = false;
  static ::ExcisionSphere<Dim> create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator,
      const std::string& excision_sphere) {
    const auto domain = domain_creator->create_domain();
    const auto& excision_spheres = domain.excision_spheres();
    if (excision_spheres.count(excision_sphere) == 0) {
      ERROR("Specified excision sphere '"
            << excision_sphere
            << "' not available. Available excision spheres are: "
            << keys_of(excision_spheres));
    }
    return excision_spheres.at(excision_sphere);
  }
};

/*!
 * \brief Triggers at which to write the coefficients of the worldtube's
 * internal Taylor series to file.
 */
struct ObserveCoefficientsTrigger : db::SimpleTag {
  using type = std::unique_ptr<Trigger>;
  using option_tags = tmpl::list<OptionTags::ObserveCoefficientsTrigger>;
  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<Trigger> create_from_options(
      const std::unique_ptr<Trigger>& trigger) {
    return deserialize<type>(serialize<type>(trigger).data());
  }
};

/*!
 * \brief The value of the scalar charge
 */
struct Charge : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::Charge>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double charge) { return charge; };
};

/*!
 * \brief The time at which the self-force is smoothly turned on.
 *
 * \details The self force is turned on using the smooth transition function
 *
 * \begin{equation}
 * w(t) = 1 - \exp{ \left(- \left(\frac{t - t_1}{\sigma} \right)^4 \right)}.
 * \end{equation}
 *
 * The turn on time is given by \f$t_1\f$.
 */
struct SelfForceTurnOnTime : db::SimpleTag {
  using type = std::optional<double>;
  using option_tags = tmpl::list<OptionTags::SelfForceOptions>;
  static constexpr bool pass_metavariables = false;
  static std::optional<double> create_from_options(
      const std::optional<OptionTags::SelfForceOptions>& self_force_options) {
    return self_force_options.has_value()
               ? std::make_optional(self_force_options->turn_on_time)
               : std::nullopt;
  };
};

/*!
 * \brief The interval over which the self-force is smoothly turned on.
 *
 * \details The self force is turned on using the smooth transition function
 *
 * \begin{equation}
 * w(t) = 1 - \exp{ \left(- \left(\frac{t - t_1}{\sigma} \right)^4 \right)}.
 * \end{equation}
 *
 * The turn on interval is given by \f$\sigma\f$.
 */
struct SelfForceTurnOnInterval : db::SimpleTag {
  using type = std::optional<double>;
  using option_tags = tmpl::list<OptionTags::SelfForceOptions>;
  static constexpr bool pass_metavariables = false;
  static std::optional<double> create_from_options(
      const std::optional<OptionTags::SelfForceOptions>& self_force_options) {
    return self_force_options.has_value()
               ? std::make_optional(self_force_options->turn_on_interval)
               : std::nullopt;
  };
};

/*!
 * \brief The mass of the scalar charge. Only has a value if the scalar self
 * force is applied.
 */
struct Mass : db::SimpleTag {
  using type = std::optional<double>;
  using option_tags = tmpl::list<OptionTags::SelfForceOptions>;
  static constexpr bool pass_metavariables = false;
  static std::optional<double> create_from_options(
      const std::optional<OptionTags::SelfForceOptions>& self_force_options) {
    return self_force_options.has_value()
               ? std::make_optional(self_force_options->mass)
               : std::nullopt;
  }
};

/*!
 * \brief The maximum number of iterations that will be applied to the
 * acceleration of the particle.
 */
struct MaxIterations : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::SelfForceOptions>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(
      const std::optional<OptionTags::SelfForceOptions>& self_force_options) {
    return self_force_options.has_value() ? self_force_options->iterations : 0;
  }
};

/*!
 * \brief The current number of iterations that has been applied to the
 * acceleration of the particle.
 */
struct CurrentIteration : db::SimpleTag {
  using type = size_t;
};

/*!
 * \brief The initial position and velocity of the scalar charge in inertial
 * coordinates.
 */
struct InitialPositionAndVelocity : db::SimpleTag {
  using type = std::array<tnsr::I<double, 3, Frame::Inertial>, 2>;
  using option_tags =
      tmpl::list<domain::OptionTags::DomainCreator<3>,
                 OptionTags::ExcisionSphere, ::OptionTags::InitialTime>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<3>>& domain_creator,
      const std::string& excision_sphere_name, const double initial_time) {
    // only evaluated at initial time, so expiration times don't matter
    const auto initial_fot = domain_creator->functions_of_time();
    const auto domain = domain_creator->create_domain();
    const auto& excision_sphere =
        domain.excision_spheres().at(excision_sphere_name);
    ASSERT(excision_sphere.is_time_dependent(),
           "excision_sphere not time dependent");
    const auto& maps = excision_sphere.moving_mesh_grid_to_inertial_map();
    const auto mapped_tuple = maps.coords_frame_velocity_jacobians(
        excision_sphere.center(), initial_time, initial_fot);
    return {std::get<0>(mapped_tuple), std::get<3>(mapped_tuple)};
  }
};

/// @{
/*!
 * \brief The position and velocity of the scalar charge particle orbiting a
 * central black hole given in inertial coordinates. This compute tag is meant
 * to be used by the elements.
 */
template <size_t Dim>
struct ParticlePositionVelocity : db::SimpleTag {
  using type = std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>;
};

template <size_t Dim>
struct ParticlePositionVelocityCompute : ParticlePositionVelocity<Dim>,
                                         db::ComputeTag {
  using base = ParticlePositionVelocity<Dim>;
  using return_type = std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>;
  using argument_tags = tmpl::list<ExcisionSphere<Dim>, ::Tags::Time,
                                   domain::Tags::FunctionsOfTime>;
  static void function(
      gsl::not_null<std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>*>
          position_velocity,
      const ::ExcisionSphere<Dim>& excision_sphere, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time);
};
/// @}

/*!
 * \brief The position of the scalar charge evolved by the worldtube singleton.
 * This tag is meant to be used by the worldtube singleton to evolve the orbit.
 */
template <size_t Dim>
struct EvolvedPosition : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};

/*!
 * \brief The velocity of the scalar charge evolved by the worldtube singleton.
 * This tag is meant to be used by the worldtube singleton to evolve the orbit.
 */
template <size_t Dim>
struct EvolvedVelocity : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};

/*!
 * \brief The position and velocity of the scalar charge particle orbiting a
 * central black hole given in inertial coordinates. This compute tag is meant
 * to be used by the worldtube singleton which evolves the position and velocity
 * according to an ODE along with the DG evolution.
 */
template <size_t Dim>
struct EvolvedParticlePositionVelocityCompute : ParticlePositionVelocity<Dim>,
                                                db::ComputeTag {
  using base = ParticlePositionVelocity<Dim>;
  using return_type = std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>;
  using argument_tags = tmpl::list<EvolvedPosition<Dim>, EvolvedVelocity<Dim>>;
  static void function(
      gsl::not_null<std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>*>
          position_velocity,
      const tnsr::I<DataVector, Dim>& evolved_position,
      const tnsr::I<DataVector, Dim>& evolved_velocity);
};

/// @{
/*!
 * \brief Computes the coordinate geodesic acceleration of the particle in the
 * inertial frame in Kerr-Schild coordinates.
 */
template <size_t Dim>
struct GeodesicAcceleration : db::SimpleTag {
  using type = tnsr::I<double, Dim, Frame::Inertial>;
};

template <size_t Dim>
struct GeodesicAccelerationCompute : GeodesicAcceleration<Dim>, db::ComputeTag {
  using base = GeodesicAcceleration<Dim>;
  using return_type = tnsr::I<double, Dim, Frame::Inertial>;
  using argument_tags = tmpl::list<
      ParticlePositionVelocity<Dim>,
      CurvedScalarWave::Tags::BackgroundSpacetime<gr::Solutions::KerrSchild>>;
  static void function(
      gsl::not_null<tnsr::I<double, Dim, Frame::Inertial>*> acceleration,
      const std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>&
          position_velocity,
      const gr::Solutions::KerrSchild& background_spacetime);
};
/// @}

/*!
 * \brief The coordinate time dilation factor of the scalar charge, i.e. the 0th
 * component of its 4-velocity.
 */
struct TimeDilationFactor : db::SimpleTag {
  using type = Scalar<double>;
};

/// @{
/*!
 * \brief A tuple of Tensors evaluated at the charge depending only the
 * background and the particle's position and velocity. These values are
 * effectively cached between different iterations of the worldtube scheme.
 */
template <size_t Dim>
struct BackgroundQuantities : db::SimpleTag {
  using type = tuples::TaggedTuple<
      gr::Tags::SpacetimeMetric<double, Dim>,
      gr::Tags::InverseSpacetimeMetric<double, Dim>,
      gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>,
      gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim>,
      Tags::TimeDilationFactor>;
};

template <size_t Dim>
struct BackgroundQuantitiesCompute : BackgroundQuantities<Dim>, db::ComputeTag {
  using base = BackgroundQuantities<Dim>;
  using return_type = tuples::TaggedTuple<
      gr::Tags::SpacetimeMetric<double, Dim>,
      gr::Tags::InverseSpacetimeMetric<double, Dim>,
      gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>,
      gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim>,
      Tags::TimeDilationFactor>;

  using argument_tags = tmpl::list<
      ParticlePositionVelocity<Dim>,
      CurvedScalarWave::Tags::BackgroundSpacetime<gr::Solutions::KerrSchild>>;
  static void function(gsl::not_null<return_type*> result,
                       const std::array<tnsr::I<double, Dim, Frame::Inertial>,
                                        2>& position_velocity,
                       const gr::Solutions::KerrSchild& background_spacetime);
};
/// @}

/// @{
/*!
 * \brief An optional that holds the coordinates of an element face abutting the
 * worldtube excision sphere. If the element does not abut the worldtube, this
 * holds std::nullopt. This tag should be in the databox of element chares. The
 * available frames are Grid and Inertial. The Centered template tag can be
 * turned on to center the coordinates around the position of the scalar
 * charge.
 */
template <size_t Dim, typename Frame, bool Centered>
struct FaceCoordinates : db::SimpleTag {
  using type = std::optional<tnsr::I<DataVector, Dim, Frame>>;
};

template <size_t Dim, typename Frame, bool Centered>
struct FaceCoordinatesCompute : FaceCoordinates<Dim, Frame, Centered>,
                                db::ComputeTag {
  using base = FaceCoordinates<Dim, Frame, Centered>;
  static constexpr bool needs_inertial_wt_coords =
      (Centered and std::is_same_v<Frame, ::Frame::Inertial>);
  using argument_tags = tmpl::flatten<
      tmpl::list<ExcisionSphere<Dim>, domain::Tags::Element<Dim>,
                 domain::Tags::Coordinates<Dim, Frame>, domain::Tags::Mesh<Dim>,
                 tmpl::conditional_t<needs_inertial_wt_coords,
                                     tmpl::list<ParticlePositionVelocity<Dim>>,
                                     tmpl::list<>>>>;

  using return_type = std::optional<tnsr::I<DataVector, Dim, Frame>>;
  static void function(
      const gsl::not_null<std::optional<tnsr::I<DataVector, Dim, Frame>>*>
          result,
      const ::ExcisionSphere<Dim>& excision_sphere, const Element<Dim>& element,
      const tnsr::I<DataVector, Dim, Frame>& coords, const Mesh<Dim>& mesh);

  static void function(
      const gsl::not_null<
          std::optional<tnsr::I<DataVector, Dim, ::Frame::Inertial>>*>
          result,
      const ::ExcisionSphere<Dim>& excision_sphere, const Element<Dim>& element,
      const tnsr::I<DataVector, Dim, ::Frame::Inertial>& coords,
      const Mesh<Dim>& mesh,
      const std::array<tnsr::I<double, Dim, ::Frame::Inertial>, 2>&
          particle_position_velocity);
};
/// @}

/// @{
/*!
 * \brief The value of the scalar field and its time derivative on element faces
 * forming the worldtube boundary, as well as the Euclidean area element of the
 * face.
 *
 * \details If the element does not abut the worldtube, this will be
 * `std::nullopt`.
 */
struct FaceQuantities : db::SimpleTag {
  using type = std::optional<Variables<tmpl::list<
      CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
      gr::surfaces::Tags::AreaElement<DataVector>>>>;
};

struct FaceQuantitiesCompute : FaceQuantities, db::ComputeTag {
  static constexpr size_t Dim = 3;
  using base = FaceQuantities;
  using return_type = std::optional<Variables<tmpl::list<
      CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
      gr::surfaces::Tags::AreaElement<DataVector>>>>;
  using tags_to_slice_to_face =
      tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                 CurvedScalarWave::Tags::Phi<Dim>,
                 gr::Tags::Shift<DataVector, Dim>, gr::Tags::Lapse<DataVector>,
                 domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>>;
  using argument_tags = tmpl::flatten<
      tmpl::list<tags_to_slice_to_face, ExcisionSphere<Dim>,
                 domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>>>;

  static void function(
      gsl::not_null<return_type*> result, const Scalar<DataVector>& psi,
      const Scalar<DataVector>& pi, const tnsr::i<DataVector, Dim>& phi,
      const tnsr::I<DataVector, Dim>& shift, const Scalar<DataVector>& lapse,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const ::ExcisionSphere<Dim>& excision_sphere, const Element<Dim>& element,
      const Mesh<Dim>& mesh);
};
/// @}

/*!
 * \brief The internal expansion order of the worldtube solution.
 */
struct ExpansionOrder : db::SimpleTag {
  using type = size_t;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::ExpansionOrder>;
  static size_t create_from_options(const size_t order) { return order; }
};

/// @{
/*!
 * Computes the puncture field on an element face abutting the worldtube
 * assuming geodesic acceleration. If the current element does not abut the
 * worldtube this holds a std::nullopt.
 */
template <size_t Dim>
struct PunctureField : db::SimpleTag {
  using type = std::optional<Variables<tmpl::list<
      CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
      ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                    Frame::Inertial>>>>;
};

template <size_t Dim>
struct PunctureFieldCompute : PunctureField<Dim>, db::ComputeTag {
  using base = PunctureField<Dim>;
  using argument_tags =
      tmpl::list<FaceCoordinates<Dim, Frame::Inertial, true>,
                 ParticlePositionVelocity<Dim>, GeodesicAcceleration<Dim>,
                 Charge, ExpansionOrder>;
  using return_type = std::optional<Variables<tmpl::list<
      CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
      ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                    Frame::Inertial>>>>;
  static void function(
      const gsl::not_null<return_type*> result,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          inertial_face_coords_centered,
      const std::array<tnsr::I<double, Dim, ::Frame::Inertial>, 2>&
          particle_position_velocity,
      const tnsr::I<double, Dim>& particle_acceleration, double charge,
      const size_t expansion_order);
};
/// @}

/*!
 * \brief Holds the current iteration of the puncture field computed with the
 * current iteration of the acceleration which includes the scalar self-force.
 * It is computed in `Actions::IteratePunctureField`.
 */
template <size_t Dim>
struct IteratedPunctureField : db::SimpleTag {
  using type = std::optional<Variables<tmpl::list<
      CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
      ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                    Frame::Inertial>>>>;
};

/*!
 * \brief A map that holds the grid coordinates centered on the worldtube of
 * all element faces abutting the worldtube with the corresponding ElementIds.
 */
template <size_t Dim>
struct ElementFacesGridCoordinates : db::SimpleTag {
  using type =
      std::unordered_map<ElementId<Dim>, tnsr::I<DataVector, Dim, Frame::Grid>>;
};

/*!
 * \brief The solution inside the worldtube, evaluated at the face coordinates
 * of an abutting element. This tag is used to provide boundary conditions to
 * the element in \ref CurvedScalarWave::BoundaryConditions::Worldtube .
 */
template <size_t Dim>
struct WorldtubeSolution : db::SimpleTag {
  using type = Variables<
      tmpl::list<::CurvedScalarWave::Tags::Psi, ::CurvedScalarWave::Tags::Pi,
                 ::CurvedScalarWave::Tags::Phi<Dim>>>;
};

/*!
 * \brief The scalar field inside the worldtube.
 *
 * \details This tag is used as a base tag for Stf::Tags::StfTensor
 */
struct PsiWorldtube : db::SimpleTag {
  using type = Scalar<double>;
};

/*!
 * \brief Holds the constant coefficient of the regular field inside the
 * worldtube.
 *
 * \details At orders n = 0 or 1 this is just equal to the monopole, but at n =
 * 2, the monopole gets an additional contribution from the trace of the second
 * order coefficient. At this point, this tag is used to solve an ODE based on
 * the expanded Klein-Gordon equation. It is implemented as a `Scalar` of size 1
 * because the evolution system does not work with doubles.
 */
struct Psi0 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Holds the time derivative of Psi0 which is used as a reduction
 * variable.
 */
struct dtPsi0 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

}  // namespace Tags
}  // namespace CurvedScalarWave::Worldtube
