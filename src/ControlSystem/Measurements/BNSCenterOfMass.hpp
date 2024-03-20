// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/RunCallbacks.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Actions/FunctionsOfTimeAreReady.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class ElementId;
template <size_t Dim>
class Mesh;
namespace control_system::Tags {
struct WriteDataToDisk;
}  // namespace control_system::Tags
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
namespace grmhd::ValenciaDivClean::Tags {
struct TildeD;
}  // namespace grmhd::ValenciaDivClean::Tags
namespace Tags {
struct PreviousTriggerTime;
}  // namespace Tags
/// \endcond

namespace control_system {
namespace measurements {
/// \cond
template <typename ControlSystems>
struct PostReductionSendBNSStarCentersToControlSystem;
/// \endcond

/*!
 * \brief  Factored Center of Mass calculation (for easier testing)
 *
 * \details This function computes the integral of tildeD (assumed to be the
 * conservative baryon density in the inertial frame), as well as its first
 * moment in the grid frame. The integrals are limited to \f$x>0\f$ (label A) or
 * \f$x<0\f$ (label B).
 *
 * \param mass_a Integral of tildeD (x > 0)
 * \param mass_b Integral of tildeD (x < 0)
 * \param first_moment_A First moment of integral of tildeD (x > 0)
 * \param first_moment_B First moment of integral of tildeD (x < 0)
 * \param mesh The mesh
 * \param inv_det_jacobian The inverse determinant of the jacobian of the map
 * between logical and inertial coordinates \param tilde_d TildeD on the mesh
 * \param x_grid The coordinates in the grid frame
 */
void center_of_mass_integral_on_element(
    const gsl::not_null<double*> mass_a, const gsl::not_null<double*> mass_b,
    const gsl::not_null<std::array<double, 3>*> first_moment_A,
    const gsl::not_null<std::array<double, 3>*> first_moment_B,
    const Mesh<3>& mesh, const Scalar<DataVector>& inv_det_jacobian,
    const Scalar<DataVector>& tilde_d,
    const tnsr::I<DataVector, 3, Frame::Grid>& x_grid);
}  // namespace measurements

/*!
 * \brief An `::Event` that computes the center of mass for $x > 0$ and $x < 0$
 * where $x$ is the in the `Frame::Grid`.
 *
 * \details See
 * `control_system::measurements::center_of_mass_integral_on_element` for the
 * calculation of the CoM. This event then does a reduction and calls
 * `control_system::PostReductionSendBNSStarCentersToControlSystem` as a post
 * reduction callback.
 *
 * \tparam ControlSystems `tmpl::list` of all control systems that use this
 * event.
 */
template <typename ControlSystems>
class BNSEvent : public ::Event {
 public:
  /// \cond
  // LCOV_EXCL_START
  explicit BNSEvent(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BNSEvent);  // NOLINT
  // LCOV_EXCL_STOP
  /// \endcond

  // This event is created during control system initialization, not
  // from the input file.
  static constexpr bool factory_creatable = false;
  BNSEvent() = default;

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Time, ::Tags::PreviousTriggerTime,
                 ::Events::Tags::ObserverMesh<3>,
                 ::Events::Tags::ObserverDetInvJacobian<Frame::ElementLogical,
                                                        Frame::Inertial>,
                 grmhd::ValenciaDivClean::Tags::TildeD,
                 ::Events::Tags::ObserverCoordinates<3, Frame::Grid>>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const double time, const std::optional<double>& previous_time,
                  const Mesh<3>& mesh,
                  const Scalar<DataVector>& inv_det_jacobian,
                  const Scalar<DataVector>& tilde_d,
                  const tnsr::I<DataVector, 3, Frame::Grid>& x_grid,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    const LinkedMessageId<double> measurement_id{time, previous_time};

    // Initialize integrals and perform local calculations
    double mass_a = 0.;
    double mass_b = 0.;
    std::array<double, 3> first_moment_a = {0., 0., 0.};
    std::array<double, 3> first_moment_b = {0., 0., 0.};
    measurements::center_of_mass_integral_on_element(
        &mass_a, &mass_b, &first_moment_a, &first_moment_b, mesh,
        inv_det_jacobian, tilde_d, x_grid);

    // Reduction
    auto my_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index];
    // We need a place to run RunCallback on... this does not need to be
    // the control system using the CoM data.
    auto& reduction_target_proxy = Parallel::get_parallel_component<
        ControlComponent<Metavariables, tmpl::front<ControlSystems>>>(cache);
    Parallel::ReductionData<
        Parallel::ReductionDatum<LinkedMessageId<double>, funcl::AssertEqual<>>,
        Parallel::ReductionDatum<double, funcl::Plus<>>,
        Parallel::ReductionDatum<double, funcl::Plus<>>,
        Parallel::ReductionDatum<std::array<double, 3>, funcl::Plus<>>,
        Parallel::ReductionDatum<std::array<double, 3>, funcl::Plus<>>>
        reduction_data{measurement_id, mass_a, mass_b, first_moment_a,
                       first_moment_b};
    Parallel::contribute_to_reduction<
        measurements::PostReductionSendBNSStarCentersToControlSystem<
            ControlSystems>>(std::move(reduction_data), my_proxy,
                             reduction_target_proxy);
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*component*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }
};

/// \cond
template <typename ControlSystems>
PUP::able::PUP_ID BNSEvent<ControlSystems>::my_PUP_ID = 0;  // NOLINT
/// \endcond

namespace measurements {
namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ControlSystemGroup
/// DataBox tag for location of neutron star center (or more accurately, center
/// of mass of the matter in the x>0 (label A) or x<0 (label B) region, in grid
/// coordinates).
template <::domain::ObjectLabel Center>
struct NeutronStarCenter : db::SimpleTag {
  using type = std::array<double, 3>;
};
}  // namespace Tags

/*!
 * \brief Measurement providing the location of the center of mass of the matter
 * in the \f$x>0\f$ and \f$x<0\f$ regions (assumed to correspond to the center
 * of mass of the two neutron stars in a BNS merger).
 *
 * \details We use Events::Tags::ObserverXXX for tags that might need to be
 * retrieved from either the Subcell or DG grid.
 */
struct BothNSCenters : tt::ConformsTo<protocols::Measurement> {
  struct FindTwoCenters : tt::ConformsTo<protocols::Submeasurement> {
    static std::string name() { return "BothNSCenters::FindTwoCenters"; }
    /// Unused tag needed to conform to the submeasurement protocol.
    template <typename ControlSystems>
    using interpolation_target_tag = void;

    template <typename ControlSystems>
    using event = BNSEvent<ControlSystems>;
  };
  /// List of submeasurements used by this measurement -- only FindTwoCenters
  /// here.
  using submeasurements = tmpl::list<FindTwoCenters>;
};

/*!
 * \brief Simple action called after reduction of the center of mass data.
 *
 * \details `mass_a`, `mass_b`, `first_moment_a`, and `first_moment_b` will
 * contain the reduced data for the integral of the density (and its first
 * moment) in the x>=0 (A label) and x<0 (B label) regions. This action
 * calculates the center of mass in each region, and sends the result to the
 * control system.
 *
 * If the `control_system::Tags::WriteDataToDisk` tag is true, then this will
 * also write the centers of mass to the `/ControlSystems/BnsCenters` subfile of
 * the reductions h5 file. The columns of the file are
 *
 * - %Time
 * - Center_A_x
 * - Center_A_y
 * - Center_A_z
 * - Center_B_x
 * - Center_B_y
 * - Center_B_z
 */
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

    if (Parallel::get<control_system::Tags::WriteDataToDisk>(cache)) {
      std::vector<double> data_to_write{
          measurement_id.id, center_a[0], center_a[1], center_a[2],
          center_b[0],       center_b[1], center_b[2]};

      auto& writer_proxy = Parallel::get_parallel_component<
          observers::ObserverWriter<Metavariables>>(cache);

      Parallel::threaded_action<
          observers::ThreadedActions::WriteReductionDataRow>(
          // Node 0 is always the writer
          writer_proxy[0], subfile_path_, legend_,
          std::make_tuple(std::move(data_to_write)));
    }
  }

 private:
  const static inline std::vector<std::string> legend_{
      "Time",       "Center_A_x", "Center_A_y", "Center_A_z",
      "Center_B_x", "Center_B_y", "Center_B_z"};
  const static inline std::string subfile_path_{"/ControlSystems/BnsCenters"};
};
}  // namespace measurements
}  // namespace control_system
