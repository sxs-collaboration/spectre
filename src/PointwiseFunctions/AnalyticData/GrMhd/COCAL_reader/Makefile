FC = gfortran
FFLAGS = -O2 -fPIC

SOURCES = \
    phys_constant.f90 \
    def_matter_parameter.f90 \
    def_quantities.f90 \
    def_bh_parameter.f90 \
    def_peos_parameter.f90 \
    def_binary_parameter.f90 \
    def_matter_parameter_mpt.f90 \
    def_binary_parameter_mpt.f90 \
    def_peos_parameter_mpt.f90 \
    make_char1_array_2d.f90 \
    make_char2_array_2d.f90 \
    make_int_array_2d.f90 \
    make_array_1d.f90 \
    make_array_2d.f90 \
    make_array_3d.f90 \
    grid_parameter.f90 \
    interface_modules_cartesian.f90 \
    coordinate_grav_r.f90 \
    coordinate_grav_phi.f90 \
    coordinate_grav_theta.f90 \
    coordinate_grav_extended.f90 \
    trigonometry_grav_theta.f90 \
    trigonometry_grav_phi.f90 \
    grid_parameter_binary_excision.f90 \
    interface_IO_input_CF_grav_export.f90 \
    interface_IO_input_WL_grav_export_hij.f90 \
    interface_IO_input_grav_export_Kij.f90 \
    interface_IO_input_CF_star_export.f90 \
    interface_invhij_WL_export.f90 \
    interface_index_vec_down2up_export.f90 \
    interface_interpo_gr2fl_metric_CF_export.f90 \
    interface_IO_input_grav_export_Ai.f90 \
    interface_IO_input_grav_export_Faraday.f90 \
    interface_IO_input_star4ve_export.f90 \
    interface_interpo_gr2fl_export.f90 \
    interface_interpo_lag4th_2Dsurf.f90 \
    interface_IO_input_CF_surf_export.f90 \
    interface_IO_input_CF_flco_export.f90 \
    interface_IO_input_CF_flir_export.f90 \
    interface_IO_input_CF_flsp_export.f90 \
    interface_IO_input_gradvep_export.f90 \
    IO_input_CF_grav_export.f90 \
    IO_input_WL_grav_export_hij.f90 \
    IO_input_grav_export_Kij.f90 \
    IO_input_CF_star_export.f90 \
    invhij_WL_export.f90 \
    index_vec_down2up_export.f90 \
    interpo_gr2fl_metric_CF_export.f90 \
    IO_input_grav_export_Ai.f90 \
    IO_input_grav_export_Faraday.f90 \
    IO_input_star4ve_export.f90 \
    interpo_gr2fl_export.f90 \
    interpo_gr2cgr_4th.f90 \
    interpo_fl2cgr_4th_export.f90 \
    interpo_lag4th_2Dsurf.f90 \
    lagint_4th.f90 \
    peos_initialize.f90 \
    peos_q2hprho.f90 \
    peos_lookup.f90 \
    copy_array_static_1dto2d_mpt.f90 \
    copy_int_array_static_1dto2d_mpt.f90 \
    copy_int_array_static_2dto3d_mpt.f90 \
    copy_array_static_2dto1d_mpt.f90 \
    copy_int_array_static_2dto1d_mpt.f90 \
    copy_int_array_static_3dto2d_mpt.f90 \
    copy_array_2dto3d_mpt.f90 \
    copy_array_3dto2d_mpt.f90 \
    copy_array_static_0dto1d_mpt.f90 \
    copy_array_static_1dto0d_mpt.f90 \
    grid_parameter_mpt.f90 \
    grid_parameter_binary_excision_mpt.f90 \
    allocate_grid_parameter_mpt.f90 \
    allocate_grid_parameter_binary_excision_mpt.f90 \
    allocate_def_matter_parameter_mpt.f90 \
    trigonometry_grav_phi_mpt.f90 \
    trigonometry_grav_theta_mpt.f90 \
    coordinate_grav_r_mpt.f90 \
    coordinate_grav_phi_mpt.f90 \
    coordinate_grav_theta_mpt.f90 \
    read_parameter_mpt.f90 \
    read_parameter_binary_excision_mpt.f90 \
    read_surf_parameter_mpt.f90 \
    coordinate_grav_extended_mpt.f90 \
    copy_grid_parameter_to_mpt.f90 \
    copy_grid_parameter_from_mpt.f90 \
    copy_grid_parameter_binary_excision_from_mpt.f90 \
    copy_grid_parameter_binary_excision_to_mpt.f90 \
    copy_coordinate_grav_extended_to_mpt.f90 \
    copy_coordinate_grav_extended_from_mpt.f90 \
    copy_coordinate_grav_phi_to_mpt.f90 \
    copy_coordinate_grav_phi_from_mpt.f90 \
    copy_coordinate_grav_r_to_mpt.f90 \
    copy_coordinate_grav_r_from_mpt.f90 \
    copy_coordinate_grav_theta_to_mpt.f90 \
    copy_coordinate_grav_theta_from_mpt.f90 \
    copy_trigonometry_grav_phi_to_mpt.f90 \
    copy_trigonometry_grav_phi_from_mpt.f90 \
    copy_trigonometry_grav_theta_to_mpt.f90 \
    copy_trigonometry_grav_theta_from_mpt.f90 \
    copy_def_binary_parameter_to_mpt.f90 \
    copy_def_peos_parameter_to_mpt.f90 \
    copy_def_peos_parameter_from_mpt.f90 \
    copy_def_binary_parameter_from_mpt.f90 \
    IO_input_CF_surf_export.f90 \
    IO_input_CF_flco_export.f90 \
    IO_input_CF_flir_export.f90 \
    IO_input_CF_flsp_export.f90 \
    peos_initialize_mpt.f90 \
    set_allocate_size_mpt.f90 \
    coordinate_patch_kit_grav_grid_coc2cac_mpt.f90 \
    allocate_trigonometry_grav_phi_mpt.f90 \
	coc2cac_bns.f90 \
    IO_input_gradvep_export.f90

OBJECTS = $(SOURCES:.f90=.o)

all: libcocal.a

libcocal.a: $(OBJECTS)
	ar rcs $@ $^

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) libcocal.a

.PHONY: all clean
