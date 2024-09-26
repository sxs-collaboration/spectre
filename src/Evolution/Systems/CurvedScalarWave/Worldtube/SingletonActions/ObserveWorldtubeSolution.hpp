// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace CurvedScalarWave::Worldtube::Actions {

/*!
 * \brief When Tags::ObserveCoefficientsTrigger is triggered, write the
 * coefficients of the Taylor expansion of the regular field as well as the
 * current particle's position, velocity and acceleration to file.
 */
struct ObserveWorldtubeSolution {
  using reduction_data = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<std::vector<double>, funcl::AssertEqual<>>>;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<reduction_data>>;
  static constexpr size_t Dim = 3;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (db::get<Tags::ObserveCoefficientsTrigger>(box).is_triggered(box) and
        db::get<::Tags::TimeStepId>(box).substep() == 0) {
      const auto& inertial_particle_position =
          db::get<Tags::EvolvedPosition<Dim>>(box);
      const auto& particle_velocity = db::get<Tags::EvolvedVelocity<Dim>>(box);
      const auto& particle_acceleration =
          db::get<::Tags::dt<Tags::EvolvedVelocity<Dim>>>(box);
      const size_t expansion_order = db::get<Tags::ExpansionOrder>(box);
      const auto& psi_monopole = db::get<
          Stf::Tags::StfTensor<Tags::PsiWorldtube, 0, Dim, Frame::Inertial>>(
          box);
      const auto& dt_psi_monopole =
          db::get<Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0, Dim,
                                       Frame::Inertial>>(box);

      // number of components in Taylor series
      const size_t num_coefs = ((expansion_order + 3) * (expansion_order + 2) *
                                (expansion_order + 1)) /
                               6;
      // offset everything by the 3 * 3 components for position, velocity and
      // acceleration
      const size_t pos_offset = 9;
      std::vector<double> psi_coefs(2 * num_coefs + pos_offset);
      for (size_t i = 0; i < 3; ++i) {
        psi_coefs[i] = inertial_particle_position.get(i)[0];
        psi_coefs[i + 3] = particle_velocity.get(i)[0];
        psi_coefs[i + 6] = particle_acceleration.get(i)[0];
      }
      psi_coefs[0 + pos_offset] = get(psi_monopole);
      psi_coefs[num_coefs + pos_offset] = get(dt_psi_monopole);
      if (expansion_order > 0) {
        const auto& psi_dipole = db::get<
            Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Inertial>>(
            box);
        const auto& dt_psi_dipole =
            db::get<Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 1, Dim,
                                         Frame::Inertial>>(box);
        for (size_t i = 0; i < Dim; ++i) {
          psi_coefs[1 + i + pos_offset] = psi_dipole.get(i);
          psi_coefs[num_coefs + 1 + i + pos_offset] = dt_psi_dipole.get(i);
        }
      }
      const auto legend = [&expansion_order]() -> std::vector<std::string> {
        switch (expansion_order) {
          case (0):
            return {"Time",           "Position_x",     "Position_y",
                    "Position_z",     "Velocity_x",     "Velocity_y",
                    "Velocity_z",     "Acceleration_x", "Acceleration_y",
                    "Acceleration_z", "Psi0",           "dtPsi0"};
            break;
          case (1):
            return {"Time",           "Position_x",     "Position_y",
                    "Position_z",     "Velocity_x",     "Velocity_y",
                    "Velocity_z",     "Acceleration_x", "Acceleration_y",
                    "Acceleration_z", "Psi0",           "Psix",
                    "Psiy",           "Psiz",           "dtPsi0",
                    "dtPsix",         "dtPsiy",         "dtPsiz"};
            break;
          default:
            ERROR("requested invalid expansion order");
        }
      }();
      const auto current_time = db::get<::Tags::Time>(box);
      const auto observation_id =
          observers::ObservationId(current_time, "/Worldtube");
      auto& reduction_writer = Parallel::get_parallel_component<
          observers::ObserverWriter<Metavariables>>(cache);

      Parallel::threaded_action<
          observers::ThreadedActions::WriteReductionDataRow>(
          reduction_writer[0], std::string{"/PsiTaylorCoefs"}, legend,
          std::make_tuple(current_time, psi_coefs));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace CurvedScalarWave::Worldtube::Actions
