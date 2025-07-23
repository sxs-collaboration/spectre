// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/StrahlkorperTransformations.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah {
/*!
 * \brief Invoke the callbacks specified in the `horizon_find_callbacks` alias
 * of the \p HorizonMetavars.
 *
 * \details Before invoking the callbacks, this function
 *
 * 1. Restricts the final interpolated variables from the $L_\mathrm{mesh}$ used
 *    for the FastFlow algorithm, to the actual $L$ of the Strahlkorper.
 * 2. Adds the current Strahlkorper to the `ah::Tags::PreviousSurfaces`.
 * 3. Copies the Strahlkorper, its time derivative, and the \p dependency into
 *    the box. Also possibly computes the Inertial coordinates of the final
 *    Strahlkorper and stores them in the box if the `frame` of the
 *    \p HorizonMetavars isn't the Inertial frame.
 */
template <typename HorizonMetavars, typename DbTags, typename Metavariables>
void invoke_callbacks(const gsl::not_null<db::DataBox<DbTags>*> box,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const std::optional<std::string>& dependency,
                      const FastFlow::Status status) {
  using Fr = typename HorizonMetavars::frame;

  const auto& current_time = db::get<ah::Tags::CurrentTime>(*box).value();
  const auto& all_storage = db::get<ah::Tags::Storage<Fr>>(*box);
  const auto& current_time_storage = all_storage.at(current_time);
  const auto& current_iteration = current_time_storage.current_iteration;
  auto& previous_surfaces =
      db::get_mutable_reference<ah::Tags::PreviousSurfaces<Fr>>(box);
  const auto& fast_flow = db::get<ah::Tags::FastFlow>(*box);

  // The interpolated variables have been interpolated from the volume to
  // the points on the prolonged_strahlkorper, not to the points on the
  // actual strahlkorper. So here we do a restriction of these quantities
  // onto the actual strahlkorper.
  const auto& current_strahlkorper = current_iteration.strahlkorper;
  const auto& current_ylm = current_strahlkorper.ylm_spherepack();
  const size_t L_mesh = fast_flow.current_l_mesh(current_strahlkorper);
  const auto prolonged_strahlkorper =
      ylm::Strahlkorper<Fr>(L_mesh, L_mesh, current_strahlkorper);
  const auto& prolonged_ylm = prolonged_strahlkorper.ylm_spherepack();
  const auto& prolonged_interpolated_vars = current_iteration.interpolated_vars;
  db::mutate<::Tags::Variables<ah::vars_to_interpolate_to_target<3, Fr>>>(
      [&](const gsl::not_null<
          ::Variables<ah::vars_to_interpolate_to_target<3, Fr>>*>
              new_interpolated_vars) {
        new_interpolated_vars->initialize(current_ylm.physical_size());
        tmpl::for_each<ah::vars_to_interpolate_to_target<3, Fr>>(
            [&]<typename Tag>(tmpl::type_<Tag>) {
              const auto& prolonged_var = get<Tag>(prolonged_interpolated_vars);
              auto& new_var = get<Tag>(*new_interpolated_vars);
              for (size_t i = 0; i < prolonged_var.size(); i++) {
                new_var[i] =
                    current_ylm.spec_to_phys(prolonged_ylm.prolong_or_restrict(
                        prolonged_ylm.phys_to_spec(prolonged_var[i]),
                        current_ylm));
              }
            });
      },
      box);

  // This is the number of previous strahlkorpers that we
  // keep around.
  const size_t num_previous_strahlkorpers = 3;

  // Update the previous strahlkorpers. We do this before the callbacks
  // in case any of the callbacks need the previous strahlkorpers with the
  // current strahlkorper already in it.
  previous_surfaces.emplace_front(current_time, current_strahlkorper);

  // Remove old previous_strahlkorpers that are no longer relevant.
  while (previous_surfaces.size() > num_previous_strahlkorpers) {
    previous_surfaces.pop_back();
  }

  db::mutate<ylm::Tags::Strahlkorper<Fr>, ylm::Tags::TimeDerivStrahlkorper<Fr>,
             ah::Tags::Dependency>(
      [&](const gsl::not_null<ylm::Strahlkorper<Fr>*> horizon,
          const gsl::not_null<ylm::Strahlkorper<Fr>*> time_deriv_of_horizon,
          const gsl::not_null<std::optional<std::string>*>
              callback_dependency) {
        *horizon = current_strahlkorper;

        // This is a hack to use ylm::time_deriv_of_strahlkorper function below
        std::deque<std::pair<double, ylm::Strahlkorper<Fr>>>
            previous_horizons{};
        for (const auto& previous_surface : previous_surfaces) {
          previous_horizons.emplace_back(previous_surface.time.id,
                                         previous_surface.surface);
        }

        *time_deriv_of_horizon = current_strahlkorper;
        ylm::time_deriv_of_strahlkorper(time_deriv_of_horizon,
                                        previous_horizons);

        *callback_dependency = dependency;
      },
      box);

  // Put inertial coords in the box if they aren't already there
  if constexpr (not std::is_same_v<Fr, Frame::Inertial>) {
    db::mutate<ylm::Tags::CartesianCoords<Frame::Inertial>>(
        [&](const gsl::not_null<tnsr::I<DataVector, 3>*> inertial_coords) {
          const auto& domain = Parallel::get<domain::Tags::Domain<3>>(cache);
          const auto& functions_of_time =
              Parallel::get<domain::Tags::FunctionsOfTime>(cache);
          strahlkorper_coords_in_different_frame(
              inertial_coords, current_strahlkorper, domain, functions_of_time,
              current_time.id);
        },
        box);
  }

  // Finally call callbacks
  tmpl::for_each<typename HorizonMetavars::horizon_find_callbacks>(
      [&]<typename Callback>(tmpl::type_<Callback>) {
        Callback::apply(*box, cache, status);
      });
}
}  // namespace ah
