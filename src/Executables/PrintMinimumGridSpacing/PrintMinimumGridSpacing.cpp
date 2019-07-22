// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>

#include "Domain/InitialElementIds.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/ParseOptions.hpp"

template <size_t Dim>
void compute_and_print_minimum_grid_spacing(std::string input_file) {
  Options<tmpl::list<OptionTags::DomainCreator<Dim, Frame::Inertial>>> options(
      "Print the minimum grid spacing between inertial coordinates of the "
      "Domain specified in the input file. The output can be used to "
      "choose appropriate time steps.");
  options.parse_file(input_file);
  const auto domain_creator =
      options.template get<OptionTags::DomainCreator<Dim, Frame::Inertial>>();
  auto domain = domain_creator->create_domain();
  double min_grid_spacing = std::numeric_limits<double>::max();
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator->initial_refinement_levels()[block.id()];
    const std::vector<ElementId<Dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    for (const auto& element_id : element_ids) {
      ElementMap<Dim, Frame::Inertial> map(element_id,
                                           block.coordinate_map().get_clone());
      Mesh<Dim> mesh(domain_creator->initial_extents()[block.id()],
                     Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
      min_grid_spacing = std::min(
          min_grid_spacing,
          minimum_grid_spacing(mesh.extents(), map(logical_coordinates(mesh))));
    }
  }
  printf("The minimum grid spacing is: %1.14e\n", min_grid_spacing);
}

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

int main(int argc, char* argv[]) {
  namespace bpo = boost::program_options;

  bpo::options_description command_line_options;
  command_line_options.add_options()("input-file", bpo::value<std::string>(),
                                     "Input file name");
  command_line_options.add_options()("dimensions", bpo::value<size_t>(),
                                     "Number of dimensions");

  bpo::command_line_parser command_line_parser(argc, argv);
  command_line_parser.options(command_line_options);
  command_line_parser.positional({});

  bpo::variables_map parsed_command_line_options;
  bpo::store(command_line_parser.run(), parsed_command_line_options);
  bpo::notify(parsed_command_line_options);

  if (parsed_command_line_options.count("input-file") == 0) {
    ERROR("No default input file name.  Pass --input-file.");
  }
  const std::string input_file =
      parsed_command_line_options["input-file"].as<std::string>();

  if (parsed_command_line_options.count("dimensions") == 0) {
    ERROR("No default number of dimensions.  Pass --dimensions.");
  }
  const size_t dimensions =
      parsed_command_line_options["dimensions"].as<size_t>();

  if (dimensions == 1) {
    compute_and_print_minimum_grid_spacing<1>(input_file);
  } else if (dimensions == 2) {
    compute_and_print_minimum_grid_spacing<2>(input_file);
  } else if (dimensions == 3) {
    compute_and_print_minimum_grid_spacing<3>(input_file);
  } else {
    ERROR("dimemsions must be 1, 2, or 3 but is " << dimensions);
  }
}
