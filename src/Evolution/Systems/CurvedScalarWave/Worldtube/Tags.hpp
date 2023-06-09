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
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/BackgroundSpacetime.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Tags.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"

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
      "orders 0, 1 and 2 are implemented"};
  static size_t upper_bound() { return 2; }
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

/// @{
/*!
 * \brief The position of the scalar charge particle orbiting a central black
 * hole given in inertial coordinates. We currently assume a circular orbit in
 * the xy-plane with radius \f$R\f$ and angular velocity \f$\omega =
 * R^{-3/2}\f$, where grid and inertial coordinates are equal at t = 0.
 *
 * Coordinate maps are only saved in Blocks at the moment. More generic
 * orbits will probably require injecting the grid-to-inertial coordinate map
 * into the ExcisionSpheres as well.
 */
template <size_t Dim>
struct InertialParticlePosition : db::SimpleTag {
  using type = tnsr::I<double, Dim, Frame::Inertial>;
};

template <size_t Dim>
struct InertialParticlePositionCompute : InertialParticlePosition<Dim>,
                                         db::ComputeTag {
  using base = InertialParticlePosition<Dim>;
  using return_type = tnsr::I<double, Dim, Frame::Inertial>;
  using argument_tags = tmpl::list<ExcisionSphere<Dim>, ::Tags::Time>;
  static void function(
      gsl::not_null<tnsr::I<double, Dim, Frame::Inertial>*> position,
      const ::ExcisionSphere<Dim>& excision_sphere, const double time);
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
                                     tmpl::list<InertialParticlePosition<Dim>>,
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
      const tnsr::I<double, Dim, ::Frame::Inertial>& particle_position);
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
 * Computes the puncture field on an element face abutting the worldtube.
 * If the current element does not abut the worldtube this holds a std::nullopt.
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
      tmpl::list<FaceCoordinates<Dim, Frame::Inertial, false>,
                 ExcisionSphere<Dim>, ::Tags::Time, ExpansionOrder>;
  using return_type = std::optional<Variables<tmpl::list<
      CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
      ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                    Frame::Inertial>>>>;
  static void function(
      const gsl::not_null<return_type*> result,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          inertial_face_coords,
      const ::ExcisionSphere<Dim>& excision_sphere, const double time,
      const size_t expansion_order);
};
/// @}

/*!
 * \brief Holds the advection term that is the scalar product of the
 * mesh velocity with the spatial derivative of the regular scalar field.
 */
template <size_t Dim>
struct RegularFieldAdvectiveTerm : db::SimpleTag {
  using type = Scalar<DataVector>;
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
