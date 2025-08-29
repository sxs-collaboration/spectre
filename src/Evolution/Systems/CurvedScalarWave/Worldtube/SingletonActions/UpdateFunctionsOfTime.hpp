// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>

#include "ControlSystem/UpdateFunctionOfTime.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magntitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/RadiusFunctions.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Worldtube.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system {
// forward declare
struct UpdateMultipleFunctionsOfTime;
}  // namespace control_system

namespace CurvedScalarWave::Worldtube::Actions {

/*!
 * \brief Updates the functions of time according to the motion of the
 * worldtube.
 *
 * \details We demand that the scalar charge is always in the center of the
 * worldtube and therefore deform the grid to the track the particle's motion.
 * In addition, the worldtube and black hole excision sphere radii are adjusted
 * according to smooth_broken_power_law.
 */
struct UpdateQuaternionFunctionsOfTime {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const double current_expiration_time = db::get<Tags::ExpirationTime>(box);
    const double time = db::get<::Tags::Time>(box);
    const auto& particle_pos_vel =
        db::get<Tags::ParticlePositionVelocity<3>>(box);
    const double& x = get<0>(particle_pos_vel[0]);
    const double& y = get<1>(particle_pos_vel[0]);
    const double& z = get<2>(particle_pos_vel[0]);
    const double& xdot = get<0>(particle_pos_vel[1]);
    const double& ydot = get<1>(particle_pos_vel[1]);
    const double& zdot = get<2>(particle_pos_vel[1]);
    const double r = hypot(x, y, z);
    const double radial_vel = (xdot * x + ydot * y + zdot * z) / r;
    const auto& excision_sphere = db::get<Tags::ExcisionSphere<3>>(box);
    const double grid_radius_particle =
        get(magnitude(excision_sphere.center()));

    const DataVector expansion_update{r / grid_radius_particle,
                                      radial_vel / grid_radius_particle};

    // Adding code to compute the quaternions and its derivative instead of just
    // the azimuthal angle for the generic case.
    const DataVector angular_update{(y * zdot - z * ydot) / square(r),
                                    (xdot * z - x * zdot) / square(r),
                                    (x * ydot - y * xdot) / square(r)};

    const auto& wt_radius_params =
        db::get<Tags::WorldtubeRadiusParameters>(box);
    const auto& bh_radius_params =
        db::get<Tags::BlackHoleRadiusParameters>(box);
    const double wt_radius_grid = excision_sphere.radius();
    const double wt_radius_inertial =
        smooth_broken_power_law(r, wt_radius_params[0], wt_radius_params[1],
                                wt_radius_params[2], wt_radius_params[3]);
    const double wt_radius_derivative = smooth_broken_power_law_derivative(
        r, wt_radius_params[0], wt_radius_params[1], wt_radius_params[2],
        wt_radius_params[3]);
    const double wt_radius_time_derivative = wt_radius_derivative * radial_vel;

    const auto& bh_excision_sphere =
        db::get<domain::Tags::Domain<3>>(box).excision_spheres().at(
            "ExcisionSphereB");
    const double bh_excision_radius_grid = bh_excision_sphere.radius();
    const double bh_excision_radius_inertial =
        smooth_broken_power_law(r, bh_radius_params[0], bh_radius_params[1],
                                bh_radius_params[2], bh_radius_params[3]);
    const double bh_excision_radius_derivative =
        smooth_broken_power_law_derivative(
            r, bh_radius_params[0], bh_radius_params[1], bh_radius_params[2],
            bh_radius_params[3]);
    const double bh_excision_radius_time_derivative =
        bh_excision_radius_derivative * radial_vel;

    const double sqrt_4_pi = sqrt(4. * M_PI);
    // The expansion map has already stretched the excision spheres and we need
    // to account for that.
    const DataVector size_a_update{
        sqrt_4_pi * (wt_radius_grid - wt_radius_inertial / expansion_update[0]),
        sqrt_4_pi *
            (wt_radius_inertial * expansion_update[1] -
             wt_radius_time_derivative * expansion_update[0]) /
            square(expansion_update[0])};

    const DataVector size_b_update{
        sqrt_4_pi * (bh_excision_radius_grid -
                     bh_excision_radius_inertial / expansion_update[0]),
        sqrt_4_pi *
            (bh_excision_radius_inertial * expansion_update[1] -
             bh_excision_radius_time_derivative * expansion_update[0]) /
            square(expansion_update[0])};

    // we set the new expiration time to 1/100th of a time step next to the
    // current time. This is small enough that it can handle rapid time step
    // decreases but large enough to avoid floating point precision issues.
    const double new_expiration_time =
        time +
        0.01 * (db::get<::Tags::Next<::Tags::TimeStepId>>(box).substep_time() -
                time);

    db::mutate<Tags::ExpirationTime>(
        [&new_expiration_time](const auto expiration_time) {
          *expiration_time = new_expiration_time;
        },
        make_not_null(&box));
    db::mutate<Tags::WorldtubeRadius>(
        [&wt_radius_inertial](const auto wt_radius) {
          *wt_radius = wt_radius_inertial;
        },
        make_not_null(&box));
    std::unordered_map<std::string, std::pair<DataVector, double>>
        all_updates{};
    all_updates["Rotation"] = std::make_pair(
        angular_update,
        new_expiration_time);  // angle and omega updates as a DataVector of
                               // size 2 (To be used for
                               // QuaternionWorldtubeFunctionOfTime)
    all_updates["Expansion"] =
        std::make_pair(expansion_update, new_expiration_time);
    all_updates["SizeA"] = std::make_pair(size_a_update, new_expiration_time);
    all_updates["SizeB"] = std::make_pair(size_b_update, new_expiration_time);

    Parallel::mutate<::domain::Tags::FunctionsOfTime,
                     control_system::UpdateMultipleFunctionsOfTime>(
        cache, current_expiration_time, all_updates);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace CurvedScalarWave::Worldtube::Actions
