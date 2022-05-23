// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/YlmSpherepack.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace intrp {
namespace callbacks {

/// \brief post_interpolation_callback that outputs
/// 2D "volume" data on a surface.
///
/// Uses:
/// - Metavariables
///   - `temporal_id`
/// - DataBox:
///   - `TagsToObserve` (each tag must be a Scalar<DataVector>)
///
/// Conforms to the intrp::protocols::PostInterpolationCallback protocol
///
/// For requirements on InterpolationTargetTag, see
/// intrp::protocols::InterpolationTargetTag
template <typename TagsToObserve, typename InterpolationTargetTag>
struct ObserveSurfaceData
    : tt::ConformsTo<intrp::protocols::PostInterpolationCallback> {
  static constexpr double fill_invalid_points_with =
      std::numeric_limits<double>::quiet_NaN();

  using const_global_cache_tags = tmpl::list<observers::Tags::SurfaceFileName>;

  template <typename DbTags, typename Metavariables, typename TemporalId>
  static void apply(const db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const TemporalId& temporal_id) {
    const Strahlkorper<Frame::Inertial>& strahlkorper =
        get<StrahlkorperTags::Strahlkorper<Frame::Inertial>>(box);
    const YlmSpherepack& ylm = strahlkorper.ylm_spherepack();
    const std::array<DataVector, 2> theta_phi = ylm.theta_phi_points();
    const DataVector& theta = theta_phi[0];
    const DataVector& phi = theta_phi[1];
    const DataVector& sin_theta = sin(theta);
    const DataVector& radius = ylm.spec_to_phys(strahlkorper.coefficients());

    // Output the inertial-frame coordinates.
    const std::string& surface_name =
        pretty_type::name<InterpolationTargetTag>();
    std::vector<TensorComponent> tensor_components{
        {surface_name + "/InertialCoordinates_x"s,
         radius * sin_theta * cos(phi) + strahlkorper.expansion_center()[0]},
        {surface_name + "/InertialCoordinates_y"s,
         radius * sin_theta * sin(phi) + strahlkorper.expansion_center()[1]},
        {surface_name + "/InertialCoordinates_z"s,
         radius * cos(theta) + strahlkorper.expansion_center()[2]}};

    // Output each tag if it is a scalar. Otherwise, throw a compile-time
    // error. This could be generalized to handle tensors of nonzero rank by
    // looping over the components, so each component could be visualized
    // separately as a scalar. But in practice, this generalization is
    // probably unnecessary, because Strahlkorpers are typically only
    // visualized with scalar quantities (used set the color at different
    // points on the surface).
    tmpl::for_each<TagsToObserve>([&box, &surface_name,
                                   &tensor_components](auto tag_v) {
      using Tag = tmpl::type_from<decltype(tag_v)>;
      static_assert(std::is_same_v<typename Tag::type, Scalar<DataVector>>,
                    "Each tag in TagsToObserve must be a Scalar<DataVector>. "
                    "This could be generalized if needed; see the code comment "
                    "above the static_assert.");
      tensor_components.push_back(
          {surface_name + "/"s + db::tag_name<Tag>(), get(get<Tag>(box))});
    });

    const std::string& subfile_path{std::string{"/"} + surface_name};
    const std::vector<size_t> extents_vector{
        {ylm.physical_extents()[0], ylm.physical_extents()[1]}};
    const std::vector<Spectral::Basis> bases_vector{
        2, Spectral::Basis::SphericalHarmonic};
    const std::vector<Spectral::Quadrature> quadratures_vector{
        {Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}};
    const observers::ObservationId& observation_id = observers::ObservationId(
        InterpolationTarget_detail::get_temporal_id_value(temporal_id),
        subfile_path + ".vol");

    auto& proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    // We call this on proxy[0] because the 0th element of a NodeGroup is
    // always guaranteed to be present.
    Parallel::threaded_action<observers::ThreadedActions::WriteVolumeData>(
        proxy[0], Parallel::get<observers::Tags::SurfaceFileName>(cache),
        std::string{"/" + surface_name}, observation_id,
        std::vector<ElementVolumeData>{{extents_vector, tensor_components,
                                        bases_vector, quadratures_vector}});
  }
};
}  // namespace callbacks
}  // namespace intrp
