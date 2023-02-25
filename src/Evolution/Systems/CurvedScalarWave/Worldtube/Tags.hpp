// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"

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
 * \brief The internal expansion order of the worldtube solution.
 */
struct ExpansionOrder {
  using type = size_t;
  static constexpr Options::String help{
      "The internal expansion order of the worldtube solution. Currently only "
      "order 0 is implemented"};
  static size_t upper_bound() { return 0; }
  using group = Worldtube;
};
}  // namespace OptionTags

/*!
 * \brief Tags related to the worldtube
 */
namespace Tags {
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
 * \brief An optional that holds the grid coordinates, centered on the
 * worldtube, of an element face abutting the worldtube excision sphere. If the
 * element does not abut the worldtube, this holds std::nullopt. This tag should
 * be in the databox of element chares.
 */
template <size_t Dim>
struct CenteredFaceCoordinates : db::SimpleTag {
  using type = std::optional<tnsr::I<DataVector, Dim, Frame::Grid>>;
};

template <size_t Dim>
struct CenteredFaceCoordinatesCompute : CenteredFaceCoordinates<Dim>,
                                        db::ComputeTag {
  using base = CenteredFaceCoordinates<Dim>;
  using argument_tags =
      tmpl::list<ExcisionSphere<Dim>, domain::Tags::Element<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Grid>,
                 domain::Tags::Mesh<Dim>>;
  using return_type = std::optional<tnsr::I<DataVector, Dim, Frame::Grid>>;
  static void function(
      const gsl::not_null<std::optional<tnsr::I<DataVector, Dim, Frame::Grid>>*>
          res,
      const ::ExcisionSphere<Dim>& excision_sphere, const Element<Dim>& element,
      const tnsr::I<DataVector, Dim, Frame::Grid>& grid_coords,
      const Mesh<Dim>& mesh);
};

/*!
 * \brief The internal expansion order of the worldtube solution.
 */
struct ExpansionOrder : db::SimpleTag {
  using type = size_t;
  using options_tag = tmpl::list<OptionTags::ExpansionOrder>;
  static size_t create_from_options(const size_t order) { return order; }
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

}  // namespace Tags
}  // namespace CurvedScalarWave::Worldtube
