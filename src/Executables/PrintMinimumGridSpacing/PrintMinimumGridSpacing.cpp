// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <iomanip>

#include "Domain/InitialElementIds.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Printf.hpp"
#include "Options/ParseOptions.hpp"

using frame = Frame::Inertial;

class DimensionOption;
struct Dimension {
  using type = DimensionOption;
  static constexpr OptionString help = {"help"};
};

class DimensionOption {
 public:
  struct Value {
    using type = size_t;
    static constexpr OptionString help = {
        "Defines a number of dimensions (1, 2, or 3)"};
  };
  using options = tmpl::list<Value>;
  static constexpr OptionString help = {
      "Defines a number of dimensions (1, 2, or 3)"};
  DimensionOption() = default;
  DimensionOption(size_t value, const OptionContext& context)
      : value_(std::move(value)) {
    if (value_ != 1 and value_ != 2 and value_ != 3) {
      PARSE_ERROR(context, "Dimemsion Value must be 1, 2, or 3 but is "
                               << value_);
    }
  }
  size_t value_{};
};

template <size_t Dim>
void compute_and_print_minimum_grid_spacing(std::string input_file) {
  Options<tmpl::list<Dimension, OptionTags::DomainCreator<Dim, frame>>> options(
      "");
  options.parse_file(input_file);

  const auto domain_creator =
      options.template get<OptionTags::DomainCreator<Dim, frame>>();
  auto domain = domain_creator->create_domain();
  double min_grid_spacing = std::numeric_limits<double>::max();
  for (const auto& block : domain.blocks()) {
    const auto initial_ref_levs =
        domain_creator->initial_refinement_levels()[block.id()];
    const std::vector<ElementId<Dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    for (const auto& element_id : element_ids) {
      ElementMap<Dim, frame> map(element_id,
                                 block.coordinate_map().get_clone());
      Mesh<Dim> mesh(domain_creator->initial_extents()[block.id()],
                     Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
      min_grid_spacing = std::min(
          min_grid_spacing,
          minimum_grid_spacing(mesh.extents(), map(logical_coordinates(mesh))));
    }
  }
  Parallel::printf("The minimum grid spacing is: %1.14e\n", min_grid_spacing);
}

std::string custom_help_msg() {
  constexpr int max_label_size = 21;
  const std::string help_text =
      "Print the minimum grid spacing between inertial coordinates of the "
      "Domain specified in the input file. The output can be used to "
      "choose appropriate time steps.";

  std::ostringstream ss;
  ss << "\n==== Description of expected options:\n" << help_text
     << "\n\nOptions:\n"
     << "  " << std::setw(max_label_size + 2) << std::left << "Dimension"
     << "Value\n"
     << "  " << std::setw(max_label_size + 2) << std::left << ""
     << "The dimensions of the desired domain (1, 2, or 3).\n\n"
     << "  " << std::setw(max_label_size + 2) << std::left << "DomainCreator"
     << "DomainCreator<Dim, Frame::Inertial>\n"
     << "  " << std::setw(max_label_size + 2) << std::left << ""
     << "The domain to create initially.\n"
     << "  " << std::setw(max_label_size + 2) << std::left << ""
     << "Dim must be equal to Dimension's Value.\n\n";

  return ss.str();
}

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

int main(int argc, char* argv[]) {
  namespace bpo = boost::program_options;

  bpo::options_description command_line_options;
  command_line_options.add_options()
      ("help,h", "Describe program options")
      ("input-file", bpo::value<std::string>(), "Input file name");

  bpo::command_line_parser command_line_parser(argc, argv);
  command_line_parser.options(command_line_options);
  command_line_parser.positional({});

  bpo::variables_map parsed_command_line_options;
  bpo::store(command_line_parser.run(), parsed_command_line_options);
  bpo::notify(parsed_command_line_options);

  if (parsed_command_line_options.count("help") != 0) {
    Parallel::printf("%s\n%s", command_line_options, custom_help_msg());
    return 0;
  }

  if (parsed_command_line_options.count("input-file") == 0) {
    ERROR("No default input file name.  Pass --input-file.");
  }
  const std::string input_file =
      parsed_command_line_options["input-file"].as<std::string>();

  // Parse and check only the Dimension option first
  YAML::Node config;

  try {
    config = YAML::LoadFile(input_file);
  }
  catch (YAML::BadFile& /*e*/) {
    ERROR("Could not open the input file " << input_file);
  }
  catch (const YAML::Exception& e) {
    ERROR("Unable to parse input file " << input_file << "\n\n" << e.msg);
  }

  if (not(config["Dimension"]["Value"])) {
    ERROR("Need to specify Dimension and its Value:\n\n" << custom_help_msg());
  }

  const size_t dimensions = config["Dimension"]["Value"].as<size_t>();

  if (dimensions == 1) {
    compute_and_print_minimum_grid_spacing<1>(input_file);
  } else if (dimensions == 2) {
    compute_and_print_minimum_grid_spacing<2>(input_file);
  } else if (dimensions == 3) {
    compute_and_print_minimum_grid_spacing<3>(input_file);
  } else {
    ERROR("Dimension Value must be 1, 2, or 3 but is " << dimensions);
  }
}
