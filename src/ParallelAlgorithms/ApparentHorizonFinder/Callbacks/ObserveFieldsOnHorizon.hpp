// Distributed under the MIT License.
// See LICENSE.txt for details.

// Tests for this file are combined with the ones for
// ObserveTimeSeriesOnHorizon.hpp in
// Test_ObserveFieldsAndTimeSeriesOnHorizon.cpp since they are so similar.

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Strahlkorper/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Protocols/Callback.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace ah ::callbacks {
/*!
 * \brief A `ah::protocols::Callback` that outputs 2D "volume" data on a
 * surface and the surface's spherical harmonic data.
 *
 * \details The columns of spherical harmonic data written take the form
 *
 * \code
 * [Time, {Frame}ExpansionCenter_x, {Frame}ExpansionCenter_y,
 * {Frame}ExpansionCenter_z, Lmax, coef(0,0), ... coef(Lmax,Lmax)]
 * \endcode
 *
 * where `coef(l,m)` refers to the strahlkorper coefficients stored and defined
 * by `ylm::Strahlkorper::coefficients() const`, and `Frame` is
 * `HorizonMetavars::frame`. It is assumed that $l_{\mathrm{max}} =
 * m_{\mathrm{max}}$.
 *
 * \note Currently, $l_{\mathrm{max}}$ for a given surface does not change over
 * the course of the simulation, which means that the total number of columns of
 * coefficients that we need to write is also constant. The current
 * implementation of writing the coefficients at one time assumes
 * $l_{\mathrm{max}}$ of a surface remains constant. If and when in the future
 * functionality for an adaptive $l_{\mathrm{max}}$ is added, the implementation
 * for writing the coefficients will need to be updated to account for this. One
 * possible way to address this is to have a known maximum $l_{\mathrm{max}}$
 * for a given surface and write all coefficients up to that maximum
 * $l_{\mathrm{max}}$.
 */
template <typename TagsToObserve, typename HorizonMetavars>
struct ObserveFieldsOnHorizon : tt::ConformsTo<ah::protocols::Callback> {
  using Fr = typename HorizonMetavars::frame;

  using const_global_cache_tags = tmpl::list<observers::Tags::SurfaceFileName>;

  template <typename DbTags, typename Metavariables>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const FastFlow::Status /*status*/) {
    const auto& current_time = db::get<ah::Tags::CurrentTime>(box).value();
    const auto& strahlkorper = get<ylm::Tags::Strahlkorper<Fr>>(box);
    const ylm::Spherepack& ylm = strahlkorper.ylm_spherepack();

    // Output the inertial-frame coordinates of the Strahlkorper.
    // Note that these coordinates are not Spherepack-evenly-distributed over
    // the inertial-frame sphere (they are Spherepack-evenly-distributed over
    // the frame sphere).
    std::vector<TensorComponent> tensor_components;
    const auto& inertial_strahlkorper_coords =
        get<ylm::Tags::CartesianCoords<::Frame::Inertial>>(box);
    tensor_components.push_back(
        {"InertialCoordinates_x"s, get<0>(inertial_strahlkorper_coords)});
    tensor_components.push_back(
        {"InertialCoordinates_y"s, get<1>(inertial_strahlkorper_coords)});
    tensor_components.push_back(
        {"InertialCoordinates_z"s, get<2>(inertial_strahlkorper_coords)});

    tmpl::for_each<TagsToObserve>([&box, &tensor_components](auto tag_v) {
      using Tag = tmpl::type_from<decltype(tag_v)>;
      const auto tag_name = db::tag_name<Tag>();
      const auto& tensor = get<Tag>(box);
      for (size_t i = 0; i < tensor.size(); ++i) {
        tensor_components.emplace_back(tag_name + tensor.component_suffix(i),
                                       tensor[i]);
      }
    });

    const std::string& surface_name = pretty_type::name<HorizonMetavars>();
    const std::string subfile_path{std::string{"/"} + surface_name};
    const std::vector<size_t> extents_vector{
        {ylm.physical_extents()[0], ylm.physical_extents()[1]}};
    const std::vector<Spectral::Basis> bases_vector{
        2, Spectral::Basis::SphericalHarmonic};
    const std::vector<Spectral::Quadrature> quadratures_vector{
        {Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}};
    const observers::ObservationId observation_id{current_time.id,
                                                  subfile_path + ".vol"};

    auto& proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    // We call this on proxy[0] because the 0th element of a NodeGroup is
    // always guaranteed to be present.
    Parallel::threaded_action<observers::ThreadedActions::WriteVolumeData>(
        proxy[0], Parallel::get<observers::Tags::SurfaceFileName>(cache),
        subfile_path, observation_id,
        std::vector<ElementVolumeData>{{surface_name, tensor_components,
                                        extents_vector, bases_vector,
                                        quadratures_vector}});

    std::vector<std::string> ylm_legend{};
    std::vector<double> ylm_data{};

    // When option LMax is present in the global cache,
    // check that the Strahlkorper resolution does not exceed it.
    // LMax sets a maximum resolution for
    // the Strahlkorper resolution (when adjusted using adaptive
    // horizon finding) and the number of (zero padded) columns to output,
    // to avoid H5 errors caused by different rows (corresponding to
    // different times) having different numbers of columns.
    // If this option is not present in the cache, skip the check and just write
    // the modes of the Strahlkorper, assuming that the Strahlkorper
    // resolution never changes.
    size_t max_l_to_write = strahlkorper.l_max();
    if constexpr (Parallel::is_in_global_cache<Metavariables, ah::Tags::LMax>) {
      const auto& max_resolution_and_output_l =
          Parallel::get<ah::Tags::LMax>(cache);
      if (UNLIKELY(max_resolution_and_output_l < strahlkorper.l_max())) {
        ERROR("The option LMax ("
              << max_resolution_and_output_l << ") is smaller than the current "
              << "Strahlkorper resolution L=" << strahlkorper.l_max()
              << ". Increase LMax or decrease "
              << "Strahlkorper resolution L to avoid truncating data.");
      }
      max_l_to_write = max_resolution_and_output_l;
    }

    ylm::fill_ylm_legend_and_data(make_not_null(&ylm_legend),
                                  make_not_null(&ylm_data), strahlkorper,
                                  current_time.id, max_l_to_write);

    const std::string ylm_subfile_name{std::string{"/"} + surface_name +
                                       "_Ylm"};

    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        proxy[0], ylm_subfile_name, std::move(ylm_legend),
        std::make_tuple(std::move(ylm_data)));
  }
};
}  // namespace ah::callbacks
