// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/RunCallbacks.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class ElementId;
template <size_t Dim>
class Mesh;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace domain::Tags {
template <size_t Dim>
struct Mesh;
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
namespace grmhd::ValenciaDivClean {
struct TildeD;
}
/// \endcond

namespace control_system::measurements {

template <typename ControlSystems>
struct PostReductionSendBNSStarCentersToControlSystem;

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for location of neutron star center
/// (or more accurately, center of mass of the matter
/// in the x>0 (label A) or x<0 (label B) region, in grid (distorted)
/// coordinates.
template <::domain::ObjectLabel Center>
struct NeutronStarCenter : db::SimpleTag {
  using type = std::array<double, 3>;
};
}  // namespace Tags

/// Factored Center of Mass calculation (for easier testing)
/// This function computes the integral of tildeD (assumed to be the
/// conservative baryon density in the inertial frame), as well as
/// its first moment in the grid frame. The integrals are limited to
/// \f$x>0\f$ (label A) or \f$x<0\f$ (label B).
/// inv_det_jacobian is the inverse determinant of the jacobian of the
///                  map between logical and inertial coordinates
/// x_distorted contains the coordinates in the distorted frame
///
/// \note The `x` position is in the Distorted frame to fit in with the control
/// system which works in the distorted frame. For BNS simulations, the
/// Distorted frame should be identical to the Grid frame (i.e. the identity
/// map), but it still needs to be there to interface with the control system.
void center_of_mass_integral_on_element(
    const gsl::not_null<double*> mass_a, const gsl::not_null<double*> mass_b,
    const gsl::not_null<std::array<double, 3>*> first_moment_A,
    const gsl::not_null<std::array<double, 3>*> first_moment_B,
    const Mesh<3>& mesh, const Scalar<DataVector>& inv_det_jacobian,
    const Scalar<DataVector>& tilde_d,
    const tnsr::I<DataVector, 3, Frame::Distorted>& x_distorted);

/// Measurement providing the location of the center of mass of the
/// matter in the \f$x>0\f$ and \f$x<0\f$ regions (assumed to correspond to the
/// center of mass of the two neutron stars in a BNS merger).
/// We use Events::Tags::ObserverXXX for tags that might need to be retrieved
/// from either the Subcell or DG grid.
struct BothNSCenters : tt::ConformsTo<protocols::Measurement> {
  struct FindTwoCenters : tt::ConformsTo<protocols::Submeasurement> {
    static std::string name() { return "BothNSCenters::FindTwoCenters"; }
    /// Unused tag needed to conform to the submeasurement protocol.
    template <typename ControlSystems>
    using interpolation_target_tag = void;

    /// Tags for the arguments to the apply function.
    using argument_tags =
        tmpl::list<Events::Tags::ObserverMesh<3>,
                   Events::Tags::ObserverDetInvJacobian<Frame::ElementLogical,
                                                        Frame::Inertial>,
                   grmhd::ValenciaDivClean::TildeD,
                   Events::Tags::ObserverCoordinates<3, Frame::Distorted>>;

    /// Calculate integrals needed for CoM computation on each element,
    /// then reduce the data.
    template <typename Metavariables, typename ParallelComponent,
              typename ControlSystems>
    static void apply(const Mesh<3>& mesh,
                      const Scalar<DataVector>& inv_det_jacobian,
                      const Scalar<DataVector>& tilde_d,
                      const tnsr::I<DataVector, 3, Frame::Distorted> x_grid,
                      const LinkedMessageId<double>& measurement_id,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const ElementId<3>& /*array_index*/,
                      const ParallelComponent* const /*meta*/,
                      ControlSystems /*meta*/) {
      // Initialize integrals and perform local calculations
      double mass_a = 0.;
      double mass_b = 0.;
      std::array<double, 3> first_moment_a = {0., 0., 0.};
      std::array<double, 3> first_moment_b = {0., 0., 0.};
      center_of_mass_integral_on_element(&mass_a, &mass_b, &first_moment_a,
                                         &first_moment_b, mesh,
                                         inv_det_jacobian, tilde_d, x_grid);

      // Reduction
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      // We need a place to run RunCallback on... this does not need to be
      // the control system using the CoM data.
      auto& reduction_target_proxy = Parallel::get_parallel_component<
          ControlComponent<Metavariables, tmpl::front<ControlSystems>>>(cache);
      Parallel::ReductionData<
          Parallel::ReductionDatum<LinkedMessageId<double>,
                                   funcl::AssertEqual<>>,
          Parallel::ReductionDatum<double, funcl::Plus<>>,
          Parallel::ReductionDatum<double, funcl::Plus<>>,
          Parallel::ReductionDatum<std::array<double, 3>, funcl::Plus<>>,
          Parallel::ReductionDatum<std::array<double, 3>, funcl::Plus<>>>
          reduction_data{measurement_id, mass_a, mass_b, first_moment_a,
                         first_moment_b};
      Parallel::contribute_to_reduction<
          PostReductionSendBNSStarCentersToControlSystem<ControlSystems>>(
          std::move(reduction_data), my_proxy, reduction_target_proxy);
    }
  };
  /// List of submeasurements used by this measurement -- only FindTwoCenters
  /// here.
  using submeasurements = tmpl::list<FindTwoCenters>;
};

/// Action called after reduction of the center of mass data.
/// mass_a, mass_b, first_moment_a, first_moment_b will contain the reduced data
/// for the integral of the density (and its first moment) in the x>=0 (A label)
/// and x<0 (B label) regions.
/// This action calculates the center of mass in each region, and sends the
/// result to the control system.
template <typename ControlSystems>
struct PostReductionSendBNSStarCentersToControlSystem {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const LinkedMessageId<double>& measurement_id,
                    const double& mass_a, const double& mass_b,
                    const std::array<double, 3>& first_moment_a,
                    const std::array<double, 3>& first_moment_b) {
    // Function called after reduction of the CoM data.
    // Calculate CoM from integrals
    std::array<double, 3> center_a = first_moment_a / mass_a;
    std::array<double, 3> center_b = first_moment_b / mass_b;
    const auto center_databox = db::create<
        db::AddSimpleTags<Tags::NeutronStarCenter<::domain::ObjectLabel::A>,
                          Tags::NeutronStarCenter<::domain::ObjectLabel::B>>>(
        center_a, center_b);
    // Send results to the control system(s)
    RunCallbacks<BothNSCenters::FindTwoCenters, ControlSystems>::apply(
        center_databox, cache, measurement_id);
  }
};

}  // namespace control_system::measurements
