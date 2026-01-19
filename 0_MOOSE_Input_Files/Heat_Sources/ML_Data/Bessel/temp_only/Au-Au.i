## This is the moving laser hear source simulation for 2 eta 
## Written on 19th of August 2024

[Mesh]
    type = GeneratedMesh
    dim = 2
    nx = 200
    ny = 100
    # nz = 5
    xmin = 0
    xmax = 1000
    ymin = 0
    ymax = 250
    # zmin = 0
    # zmax = 10
    # uniform_refine = 2
    elem_type = QUAD9
    
[]

[ICs]

    [velocity_x]
        variable = vel_x
        type = FunctionIC
        function =  'if(y>200&y<=250 & x>=100&x<=150, 1e-2, 1e-6)' 
    []

    [velocity_y]
        variable = vel_y
        type = FunctionIC
        function =  'if(y>200&y<=250 & x>=100&x<=150, 1e-2, 1e-6)' 
    []

[]

[BCs]

    # 1D: left = 0, right = 1
    # 2D: bottom = 0, right = 1, top = 2, left = 3
    # 3D: back = 0, bottom = 1, right = 2, top = 3, left = 4, front = 5

    # [temp_fixed]
    #     type = ADDirichletBC
    #     variable = temp
    #     boundary = 'bottom'
    #     value = 300
    # []

    [convectiveFlux_air]
        type = ConvectiveHeatFluxBC
        variable = temp
        boundary = 'left top'
        T_infinity = 300.0
        heat_transfer_coefficient = 0.05 # 50 W/m^2K for air  https://doi.org/10.1533/978-1-78242-164-1.353
        heat_transfer_coefficient_dT = 0
    []

    [convectiveFlux_left]
        type = ConvectiveHeatFluxBC
        variable = temp
        boundary = 'left'
        T_infinity = 300.0
        heat_transfer_coefficient = 1150 # 11500 W/m^2K for metal  https://doi.org/10.1016/j.intermet.2017.11.021
        heat_transfer_coefficient_dT = 0
    []

    [convectiveFlux_right]
        type = ConvectiveHeatFluxBC
        variable = temp
        boundary = 'right'
        T_infinity = 300.0
        heat_transfer_coefficient = 1150 # 11500 W/m^2K for metal  https://doi.org/10.1016/j.intermet.2017.11.021
        heat_transfer_coefficient_dT = 0
    []

    [convectiveFlux_metal]
        type = ConvectiveHeatFluxBC
        variable = temp
        boundary = 'bottom'
        T_infinity = 300.0
        heat_transfer_coefficient = 1.15e6 # 11500 W/m^2K for metal  https://doi.org/10.1016/j.intermet.2017.11.021
        heat_transfer_coefficient_dT = 0
    []


    [vel_x_bottom]
        type = ADDirichletBC
        variable = vel_x
        boundary = 'left bottom right'
        value = 0
    []

    [vel_y_bottom]
        type = ADDirichletBC
        variable = vel_y
        boundary = 'left bottom right'
        value = 0
    []

    # [Periodic]
    #     [horizontally]
    #         auto_direction = 'x'
    #         variable = 'temp'
    #     []
    # []

[]


[Variables]

    [vel_x]
        order = SECOND
        family = LAGRANGE
    []

    [vel_y]
        order = SECOND
        family = LAGRANGE
    []
    
    [p]
        order = FIRST
        family = LAGRANGE
    []

    [temp]
        initial_condition = 300
        scaling = 1.0e-10
    []

[]    

[Functions]
    [path_x]
        type = ParsedFunction
        expression = 125+30.0*t
    []

    [path_y]
        type = ParsedFunction
        expression = 250
    []

    [laser_switch]
        type = ParsedFunction
        expression = 'if(t<=25, 1, 0)'
    []
[]

[Materials]
    [scale]
        type = GenericConstantMaterial
        prop_names = 'length_scale time_scale energy_scale v_mol'  # 6.24150943e18 ev
        prop_values = '1.0e6 1.0e0 1.0e9 10.21e-6' 
    []

    # [constants]
    #     type = GenericConstantMaterial
    #     prop_names = 'sigma delta gamma R'  #sigma -> J/m^2; delta -> meter; M_si unit-> m^5/Js
    #     prop_values = '0.5 10.0e-6 1.5 8.31'
    # []

    # [M1] # M_si unit-> m^5/Js
    #     type = DerivativeParsedMaterial
    #     property_name = M1
    #     material_property_names = 'R'
    #     constant_names = 'F_M1 M01 Q1'
    #     constant_expressions = '1e-8 3.15e-07  1073.15'
    #     coupled_variables = 'temp'
    #     expression = 'F_M1*M01*exp(-Q1/(R*temp))'
    # []

    # [M2]
    #     type = DerivativeParsedMaterial
    #     property_name = M2
    #     material_property_names = 'R'
    #     constant_names = 'F_M2 M02 Q2'
    #     constant_expressions = '1e-8 3.38e-09  1165.84'
    #     coupled_variables = 'temp'
    #     expression = 'F_M2*M02*exp(-Q2/(R*temp))'
    # []

    # [L1-2]
    #     type = ParsedMaterial
    #     property_name = L1_2
    #     constant_names = factor_L
    #     constant_expressions = '1.0e8'
    #     material_property_names = 'M1 M2 length_scale time_scale energy_scale mu kappa'
    #     expression = 'factor_L*(0.5/0.2)*(M1+M2)*(mu/kappa)*(length_scale^3/(time_scale*energy_scale))'
    # []

    [switching_function]
        type = ParsedMaterial
        f_name = h
        args = 'temp'
        function =  '1/(1+exp(-10*((temp/1337)-1)))'
    []

    [mu_values]
        type = GenericConstantMaterial
        prop_names = 'pseudo_mu1       mu2'  # mu_m ==> mushy zone
        prop_values = '9.33743418e-04  3.11e-0' 
    []

    [mu_LIQ]
        type = ParsedMaterial
        property_name = mu1
        material_property_names = 'pseudo_mu1'
        constant_names = 'Q_mu'
        constant_expressions = '2200'
        coupled_variables = 'temp'
        expression = 'pseudo_mu1*exp(Q_mu/temp)' # https://link.springer.com/article/10.1007/s10765-016-2104-7
    []

    [mu_NS]
        type = ParsedMaterial
        property_name = mu_name  # To distinguish with phase field mu (https://github.com/idaholab/moose/blob/next/modules/navier_stokes/src/kernels/INSBase.C)
        material_property_names = 'length_scale time_scale energy_scale mu1 mu2 h'
        expression = '(h*mu1 + 10*mu1*h*(1-h)+ (1-h)*mu2) / (length_scale*time_scale)'
    [] 

    [conductivity_values] # Thermal conductivity of 2 phases
        type = GenericConstantMaterial
        prop_names  = 'pseudo_k1   pseudo_k2'
        prop_values = '100         338.91'      # https://doi.org/10.1016/j.jestch.2023.101413
    []

    [conductivity_LIQUID] # Function for themperature dependent thermal conductivity of LIQUID Phase
        type = DerivativeParsedMaterial
        property_name = k1
        material_property_names = 'pseudo_k1'
        constant_names =        'f_k1    alpha_k1'
        constant_expressions =  '1       0.027397'                  # Value of alpha_k3 = 0.027397 W/mK^2 reference: https://github.com/anilkunwar/temperature_dependent_material_properties/blob/main/getAuThermalConductivity.F90
        coupled_variables = 'temp'
        expression = 'f_k1*(pseudo_k1 + alpha_k1*temp)'
    []

    [conductivity_FCC]  # Function for themperature dependent thermal conductivity of FCC Phase
        type = DerivativeParsedMaterial
        property_name = k2
        material_property_names = 'pseudo_k2'
        constant_names =        'f_k2      alpha_k2'
        constant_expressions =  '1   -6.93e-2'                      # Value of alpha_k4 = -6.93e-2 W/mK reference: https://github.com/anilkunwar/temperature_dependent_material_properties/blob/main/getAuThermalConductivity.F90
        coupled_variables = 'temp'
        expression = 'f_k2*(pseudo_k2 + alpha_k2*temp)'
    []

    [conductivity]
        type = ParsedMaterial
        property_name = thermal_conductivity
        material_property_names = ' length_scale time_scale energy_scale k1 k2 h'
        expression = '(h*k1 + (1-h)*k2)*energy_scale/(length_scale*time_scale)'
    []

    [density_values] # Density of phases
        type = GenericConstantMaterial
        prop_names =    'pseudo_rho1     pseudo_rho2'
        prop_values =   '19325.28        19657.6'             # https://doi.org/10.1016/j.jestch.2023.101413
    []

    [density_LIQUID] # Function for themperature dependent density of LIQUID Phase
        type = DerivativeParsedMaterial
        property_name = rho1
        material_property_names = 'pseudo_rho1'
        constant_names =        'f_density1   alpha_rho1'
        constant_expressions =  '1              -1.44'                       # Value of alpha_rho3 = -1.44 reference: https://github.com/anilkunwar/temperature_dependent_material_properties/blob/main/getAuDensity.F90
        coupled_variables = 'temp'
        expression = 'f_density1*(pseudo_rho1 + alpha_rho1*temp)'
    []

    [density_FCC] # Function for themperature dependent density of FCC Phase
        type = DerivativeParsedMaterial
        property_name = rho2
        material_property_names = 'pseudo_rho2'
        constant_names = 'fos_density2 alpha_rho2'
        constant_expressions = '1 -1.2'                                     # Value of alpha_rho4 = -1.2 reference: https://github.com/anilkunwar/temperature_dependent_material_properties/blob/main/getAuDensity.F90
        coupled_variables = 'temp'
        expression = 'fos_density2*(pseudo_rho2 + alpha_rho2*temp)'
    []

    [density]
        type = ParsedMaterial
        property_name = density_name
        material_property_names = 'length_scale rho1 rho2 h'
        expression = '(h*rho1 + (1-h)*rho2)/(length_scale^3)'
    []

    [spec_heat_values] # Specific Heat of phases
        type = GenericConstantMaterial
        prop_names =  'pseudo_sp1  pseudo_sp2'
        prop_values = '158         132'                           # https://webbook.nist.gov/cgi/inchi?ID=C7440326&Mask=2&Type=JANAFS&Table=on (convert J/molK to J/kgK) &  https://doi.org/10.1016/S0364-5916(01)00026-8
    []

    [Sp_LIQUID] # Function for themperature dependent Specific heat of LIQUID Phase
        type = DerivativeParsedMaterial
        property_name = sp1
        material_property_names = 'pseudo_sp1'
        constant_names =       'f_sp1   alpha_sp1   beta_sp1'
        constant_expressions = '1       5.08e7      -0.0114'                           # Fitted the data from: .TDB file from https://doi.org/10.1016/S0364-5916(01)00026-8
        coupled_variables = 'temp'
        expression = 'f_sp1*(alpha_sp1*exp(beta_sp1*temp)+pseudo_sp1)'
    []

    [Sp_FCC] # Function for themperature dependent Specific heat of FCC Phase
        type = DerivativeParsedMaterial
        property_name = sp2
        material_property_names = 'pseudo_sp2'
        constant_names =        'f_sp2  alpha_sp2   beta_sp2'
        constant_expressions =  '1      2.5e-5      -0.011'                            # Fitted the data from: .TDB file from https://doi.org/10.1016/S0364-5916(01)00026-8
        coupled_variables = 'temp'
        expression = 'f_sp2*(alpha_sp2*temp^2 + beta_sp2*temp + pseudo_sp2)'
    []

    [specific_heat]
        type = ParsedMaterial
        property_name = specific_heat
        material_property_names = 'energy_scale sp1 sp2 h'
        expression = '(h*sp1 + (1-h)*sp2)*energy_scale'
    []

    [absorptivity_value]
        type = ParsedMaterial
        property_name = absorptivity
        material_property_names = 'length_scale'
        expression = '8.5e7/length_scale'
    []

    [Inner_Gaussian_Beam_Radius]
        type = ParsedMaterial
        property_name = rG
        material_property_names = 'length_scale'
        expression = '22.65e-6*length_scale'
    []

    [First_Ring_Beam_Radius]
        type = ParsedMaterial
        property_name = rR1
        material_property_names = 'length_scale'
        expression = '40.0e-6*length_scale' 
    []

    [First_ring_Beam_half_Thickness]
        type = ParsedMaterial
        property_name = rT1
        material_property_names = 'length_scale'
        expression = '30.0e-6*length_scale' 
    []

    [Second_Ring_Beam_Radius]
        type = ParsedMaterial
        property_name = rR2
        material_property_names = 'length_scale'
        expression = '70.0e-6*length_scale' 
    []

    [Second_ring_Beam_half_Thickness]
        type = ParsedMaterial
        property_name = rT2
        material_property_names = 'length_scale'
        expression = '20.0e-6*length_scale' 
    []

    [laser_power]
        type = ParsedMaterial
        property_name = Power
        material_property_names = 'energy_scale time_scale'
        expression = '250*energy_scale/time_scale'
    []

    [volumetric_heat]
        type = BesselHS
        power = Power
        efficiency = 0.75
        a0 = 0.1 # gaussian_power_prop
        a1 = 0.5 # first_ring_power_prop
        a2 = ${fparse 1-a0-a1} # second_ring_power_prop
        Ca = 2.0 # Coefficient Constant Outside Exponential
        Cb = 2.0 # Coefficient Constant Inside Exponential
        rG = rG
        rR1 = rR1
        rT1 = rT1
        rR2 = rR2
        rT2 = rT2               
        factor = 1.0e-4
        alpha = absorptivity
        function_x= path_x
        function_y= path_y
        laser_switch = laser_switch # Laser switch 1 for ON and 0 for OFF, as a Function of simulation time defined in Function bloc
    []   
[]    


[Kernels]

    [mass]
        type = INSMass
        variable = p
        u = vel_x
        v = vel_y
        pressure = p
        mu_name = mu_name
        rho_name = density_name
    []

    [x_momentum_space]
        type = INSMomentumLaplaceForm
        variable = vel_x
        u = vel_x
        v = vel_y
        pressure = p
        mu_name = mu_name
        rho_name = density_name
        gravity = '0 -9.81e+5 0'
        integrate_p_by_parts = true
        supg = true
        component = 0
    []

    [y_momentum_space]
        type = INSMomentumLaplaceForm
        variable = vel_y
        u = vel_x
        v = vel_y
        pressure = p
        mu_name = mu_name
        rho_name = density_name
        gravity = '0 -9.81e+5 0'
        integrate_p_by_parts = true
        supg = true
        component = 1
    []

    [x_momentum_time]
        type = INSMomentumTimeDerivative
        variable = vel_x 
        rho_name = density_name
    []
    
    [y_momentum_time]
        type = INSMomentumTimeDerivative
        variable = vel_y
        rho_name = density_name
    []

    [temperature_time]
        type = INSTemperatureTimeDerivative
        variable = temp
        cp_name = specific_heat
        rho_name = density_name
    []

    # [temperature_space] ## This class computes the residual and Jacobian contributions for the incompressible Navier-Stokes temperature (energy) equation.
    #     type = INSTemperature
    #     variable = temp
    #     u = vel_x
    #     v = vel_y
    #     rho_name = density_name
    #     k_name = thermal_conductivity
    #     cp_name = specific_heat
    # []

    [time]
        type = HeatConductionTimeDerivative
        variable = temp
        density_name = density_name
    []

    [heat_conduct]
        type = HeatConduction
        variable = temp
        diffusion_coefficient = thermal_conductivity
    []

    [heat_source]
        type = ADMatHeatSource
        material_property = volumetric_heat
        variable = temp
    []
   
[]



[Executioner]
    type = Transient
    solve_type          = 'PJFNK'

    # petsc_options_iname = '-pc_type -pc_hypre_type -ksp_gmres_restart -pc_hypre_boomeramg_strong_threshold'
    # petsc_options_value = 'hypre    boomeramg      31       0.7'
    
    petsc_options       = '-snes_converged_reason -ksp_converged_reason -options_left'
    petsc_options_iname = '-ksp_gmres_restart -pc_factor_shift_type -pc_factor_shift_amount -pc_type'
    petsc_options_value = '100 NONZERO 1e-15 ilu'

    l_max_its           = 30
    nl_max_its          = 50
    l_tol               = 1e-04
    nl_rel_tol          = 1e-08
    nl_abs_tol          = 1e-09

    end_time            = 28
    dt                  = 0.06

    # [Adaptivity]
    #     initial_adaptivity = 1
    #     refine_fraction = 0.7
    #     coarsen_fraction = 0.1
    #     max_h_level = 1
    #     # weight_names = 'eta1 eta2'
    #     # weight_values = '1 1'
    # []

[]

[Preconditioning]

    [SMP]
        type = SMP
        full = true
        petsc_options_iname = '-pc_type -pc_factor_shift_type -pc_factor_mat_solver_type'
        petsc_options_value = 'lu       NONZERO               strumpack'
    []

    active = 'full'

    [full]
        type = SMP
        full = true
    []

    [mydebug]
        type = FDP
        full = true
    []
[]


[Postprocessors]

    [temp_max]
        type = ElementExtremeValue
        variable = temp
    []

    [temp_avg]
        type = ElementAverageValue
        variable = temp
    []

    [temp_min]
        type = ElementExtremeValue
        variable = temp
        value_type = min
    []
[]

[Outputs]
   exodus = true
   interval = 1
   file_base = exodus/Au-Au
   csv = true
   [my_checkpoint]
       type = Checkpoint
       num_files = 2
       interval = 2
       file_base = exodus/Au-Au
   []
[]

[Debug]
    show_var_residual_norms = true
[]