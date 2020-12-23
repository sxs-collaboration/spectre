// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ScriPlusInterpolationManager.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/Observer/WriteSimpleData.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace Actions {
namespace detail {
// Provide a nicer name for the output h5 files for some of the uglier
// combinations we need
template <typename Tag>
struct ScriOutput {
  static std::string name() noexcept { return db::tag_name<Tag>(); }
};
template <typename Tag>
struct ScriOutput<Tags::ScriPlus<Tag>> {
  static std::string name() noexcept { return pretty_type::short_name<Tag>(); }
};
template <>
struct ScriOutput<Tags::Du<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>>> {
  static std::string name() noexcept { return "Psi4"; }
};

using weyl_correction_list =
    tmpl::list<Tags::Du<Tags::TimeIntegral<Tags::ScriPlus<Tags::Psi4>>>,
               Tags::ScriPlus<Tags::Psi3>, Tags::ScriPlus<Tags::Psi2>,
               Tags::ScriPlus<Tags::Psi1>, Tags::ScriPlus<Tags::Psi0>,
               Tags::EthInertialRetardedTime>;

void correct_weyl_scalars_for_inertial_time(
    gsl::not_null<Variables<weyl_correction_list>*>
        weyl_correction_variables) noexcept;
}  // namespace detail

/*!
 * \ingroup ActionsGroup
 * \brief Checks the interpolation managers and if they are ready, performs the
 * interpolation and sends the data to file via
 * `observers::ThreadedActions::WriteSimpleData`.
 *
 * \details This uses the `ScriPlusInterpolationManager` to perform the
 * interpolations of all requested scri quantities (determined by
 * `scri_values_to_observe` in the metavariables), and write them to disk using
 * `observers::threadedActions::WriteSimpleData`.
 *
 * \note This action also uses the `Tags::EthInertialRetardedTime`, interpolated
 * to the inertial frame, to perform the coordinate transformations presented in
 * \cite Boyle:2015nqa to the Weyl scalars after interpolation. For our
 * formulas, we need to adjust the signs and factors of two to be compatible
 * with our definitions of \f$\eth\f$ and choice of Newman-Penrose tetrad.
 *
 * \f{align*}{
 * \Psi_0^{\prime (5)}
 * =&  \Psi_0^{(5)} + 2 \eth u^\prime \Psi_1^{(4)}
 * + \frac{3}{4} \left(\eth u^\prime\right)^2 \Psi_2^{(3)}
 * + \frac{1}{2} \left( \eth u^\prime\right)^3  \Psi_3^{(2)}
 * + \frac{1}{16} \left(\eth u^\prime\right)^4 \Psi_4^{(1)}, \\
 * \Psi_1^{\prime (4)}
 * =&  \Psi_1^{(4)} + \frac{3}{2} \eth u^\prime \Psi_2^{(3)}
 * + \frac{3}{4} \left(\eth u^\prime\right)^2  \Psi_3^{(2)}
 * + \frac{1}{8} \left(\eth u^\prime\right)^3 \Psi_4^{(1)}, \\
 * \Psi_2^{\prime (3)}
 * =&  \Psi_2^{(3)}
 * + \eth u^\prime  \Psi_3^{(2)}
 * + \frac{1}{4} \left(\eth  u^\prime\right)^2 \Psi_4^{(1)}, \\
 * \Psi_3^{\prime (2)}
 * =& \Psi_3^{(2)} + \frac{1}{2} \eth u^{\prime} \Psi_4^{ (1)}, \\
 * \Psi_4^{\prime (1)}
 * =& \Psi_4^{(1)}.
 * \f}
 *
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: `InterpolagionManager<ComplexDataVector, Tag>` for each `Tag` in
 * `Metavariables::scri_values_to_observe`
 */
template <typename ObserverWriterComponent>
struct ScriObserveInterpolated {
  using const_global_cache_tags = tmpl::list<Tags::ObservationLMax>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const size_t observation_l_max = db::get<Tags::ObservationLMax>(box);
    const size_t l_max = db::get<Tags::LMax>(box);
    std::vector<double> data_to_write(2 * square(observation_l_max + 1) + 1);
    ComplexModalVector goldberg_modes{square(l_max + 1)};
    std::vector<std::string> file_legend;
    file_legend.reserve(2 * square(observation_l_max + 1) + 1);
    file_legend.emplace_back("time");
    for (int i = 0; i <= static_cast<int>(observation_l_max); ++i) {
      for (int j = -i; j <= i; ++j) {
        file_legend.push_back(MakeString{} << "Real Y_" << i << "," << j);
        file_legend.push_back(MakeString{} << "Imag Y_" << i << "," << j);
      }
    }
    // alternative for the coordinate transformation getting scri+ values of the
    // weyl scalars:
    // need to obtain the eth of the inertial retarded time, each of the Weyl
    // scalars, and then we'll perform a transformation on that temporary
    // variables object, then output.
    // it won't be as general, but that's largely fine. The main frustration is
    // the loss of precision.
    Variables<detail::weyl_correction_list> corrected_scri_plus_weyl{
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

    while (
        db::get<Tags::InterpolationManager<
            ComplexDataVector,
            tmpl::front<typename Metavariables::scri_values_to_observe>>>(box)
            .first_time_is_ready_to_interpolate()) {
      // first get the weyl scalars and correct them
      double interpolation_time = 0.0;
      tmpl::for_each<detail::weyl_correction_list>([&interpolation_time,
                                                    &corrected_scri_plus_weyl,
                                                    &box](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        std::pair<double, ComplexDataVector> interpolation;
        db::mutate<Tags::InterpolationManager<ComplexDataVector, tag>>(
            make_not_null(&box),
            [&interpolation](
                const gsl::not_null<
                    ScriPlusInterpolationManager<ComplexDataVector, tag>*>
                    interpolation_manager) {
              interpolation =
                  interpolation_manager->interpolate_and_pop_first_time();
            });
        interpolation_time = interpolation.first;
        get(get<tag>(corrected_scri_plus_weyl)).data() = interpolation.second;
      });

      detail::correct_weyl_scalars_for_inertial_time(
          make_not_null(&corrected_scri_plus_weyl));

      // then output each of them
      tmpl::for_each<detail::weyl_correction_list>(
          [&data_to_write, &corrected_scri_plus_weyl, &interpolation_time,
           &file_legend, &observation_l_max, &l_max, &cache,
           &goldberg_modes](auto tag_v) noexcept {
            using tag = typename decltype(tag_v)::type;
            if constexpr (tmpl::list_contains_v<
                              typename Metavariables::scri_values_to_observe,
                              tag>) {
              ScriObserveInterpolated::transform_and_write<
                  tag, tag::type::type::spin>(
                  get(get<tag>(corrected_scri_plus_weyl)).data(),
                  interpolation_time, make_not_null(&goldberg_modes),
                  make_not_null(&data_to_write), file_legend, l_max,
                  observation_l_max, cache);
            }
          });

      // then do the interpolation and output of each of the rest of the tags.
      tmpl::for_each<
          tmpl::list_difference<typename Metavariables::scri_values_to_observe,
                                detail::weyl_correction_list>>(
          [&box, &data_to_write, &file_legend, &observation_l_max, &l_max,
           &cache, &goldberg_modes](auto tag_v) noexcept {
            using tag = typename decltype(tag_v)::type;
            std::pair<double, ComplexDataVector> interpolation;
            db::mutate<Tags::InterpolationManager<ComplexDataVector, tag>>(
                make_not_null(&box),
                [&interpolation](
                    const gsl::not_null<
                        ScriPlusInterpolationManager<ComplexDataVector, tag>*>
                        interpolation_manager) {
                  interpolation =
                      interpolation_manager->interpolate_and_pop_first_time();
                });
            ScriObserveInterpolated::transform_and_write<tag,
                                                         tag::type::type::spin>(
                interpolation.second, interpolation.first,
                make_not_null(&goldberg_modes), make_not_null(&data_to_write),
                file_legend, l_max, observation_l_max, cache);
          });
    }
    return std::forward_as_tuple(std::move(box));
  }

 private:
  template <typename Tag, int Spin, typename Metavariables>
  static void transform_and_write(
      const ComplexDataVector& data, const double time,
      const gsl::not_null<ComplexModalVector*> goldberg_mode_buffer,
      const gsl::not_null<std::vector<double>*> data_to_write_buffer,
      const std::vector<std::string>& legend, const size_t l_max,
      const size_t observation_l_max,
      Parallel::GlobalCache<Metavariables>& cache) noexcept {
    const SpinWeighted<ComplexDataVector, Spin> to_transform;
    make_const_view(make_not_null(&to_transform.data()), data, 0, data.size());
    SpinWeighted<ComplexModalVector, Spin> goldberg_modes;
    goldberg_modes.set_data_ref(goldberg_mode_buffer);
    Spectral::Swsh::libsharp_to_goldberg_modes(
        make_not_null(&goldberg_modes),
        Spectral::Swsh::swsh_transform(l_max, 1, to_transform), l_max);

    (*data_to_write_buffer)[0] = time;
    for (size_t i = 0; i < square(observation_l_max + 1); ++i) {
      (*data_to_write_buffer)[2 * i + 1] = real(goldberg_modes.data()[i]);
      (*data_to_write_buffer)[2 * i + 2] = imag(goldberg_modes.data()[i]);
    }
    auto observer_proxy =
        Parallel::get_parallel_component<ObserverWriterComponent>(
            cache)[static_cast<size_t>(sys::my_node())];
    Parallel::threaded_action<observers::ThreadedActions::WriteSimpleData>(
        observer_proxy, legend, *data_to_write_buffer,
        "/" + detail::ScriOutput<Tag>::name());
  }
};
}  // namespace Actions
}  // namespace Cce
