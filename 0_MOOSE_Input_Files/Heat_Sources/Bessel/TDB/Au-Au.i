## This is the moving laser hear source simulation for 2 eta 
## Written on 12th of May 2024

[Mesh]
    type = GeneratedMesh
    dim = 2
    nx = 150 #375
    ny = 67  #100
    # nz = 5
    xmin = 0
    xmax = 750
    ymin = 0
    ymax = 201
    # zmin = 0
    # zmax = 10
    # uniform_refine = 2
    elem_type = QUAD9
    
[]

[ICs]
    [eta1]
        variable = eta1
        type = FunctionIC
        function = 'if(y>150&y<=201 & x>=70&x<=130, 1, 0)'
    []

    [velocity_x]
        variable = vel_x
        type = FunctionIC
        function =  'if(y>150&y<=201 & x>=70&x<=130, 1e-2, 1e-6)' 
    []

    [velocity_y]
        variable = vel_y
        type = FunctionIC
        function =  'if(y>150&y<=201 & x>=70&x<=130, 1e-2, 1e-6)' 
    []

[]

[BCs]

    # 1D: left = 0, right = 1
    # 2D: bottom = 0, right = 1, top = 2, left = 3
    # 3D: back = 0, bottom = 1, right = 2, top = 3, left = 4, front = 5

    [temp_fixed]
        type = ADDirichletBC
        variable = temp
        boundary = 'bottom'
        value = 300
    []

    [convectiveFlux]
        type = ConvectiveHeatFluxBC
        variable = temp
        boundary = '0'
        T_infinity = 300.0
        heat_transfer_coefficient = 0.05 # 50 W/m^2K for air  https://doi.org/10.1533/978-1-78242-164-1.353
        heat_transfer_coefficient_dT = 0
    []

    [neumann1]
        type = NeumannBC
        boundary = 'bottom'
        variable = 'eta1'
        value = 0
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
    #         # variable = 'temp'
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

    [eta1]
        order = FIRST
        family = LAGRANGE
        scaling = 1.0e-4
    []

[]    

[Functions]
    [path_x]
        type = ParsedFunction
        expression = 100+30.0*t
    []

    [path_y]
        type = ParsedFunction
        expression = 201
    []

[]

[Materials]
    [scale]
        type = GenericConstantMaterial
        prop_names = 'length_scale time_scale energy_scale v_mol'  # 6.24150943e18 ev
        prop_values = '1.0e6 1.0e0 1.0e9 10.21e-6' 
    []

    [constants]
        type = GenericConstantMaterial
        # prop_names = 'sigma delta gamma pseudo_M1 pseudo_M2 R' #sigma -> J/m^2; delta -> meter; M_si unit-> m^5/Js
        prop_names = 'sigma delta gamma M1 M2 R'
        # prop_values = '0.5 10.0e-7 1.5 1.5e-9 2.0e-14 8.31'
        prop_values = '0.5 10.0e-7 1.5 2.0e-9 2.0e-14 8.31'
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
        material_property_names = 'length_scale time_scale energy_scale mu1 mu2 h1'
        expression = '(h1*mu1 + 10*mu1*h1*(1-h1)+ (1-h1)*mu2) / (length_scale*time_scale)'
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
        material_property_names = ' length_scale time_scale energy_scale k1 k2 h1'
        expression = '(h1*k1 + (1-h1)*k2)*energy_scale/(length_scale*time_scale)'
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
        material_property_names = 'length_scale rho1 rho2 h1'
        expression = '(h1*rho1 + (1-h1)*rho2)/(length_scale^3)'
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
        material_property_names = 'energy_scale sp1 sp2 h1'
        expression = '(h1*sp1 + (1-h1)*sp2)*energy_scale'
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
        expression = '35e-6*length_scale'
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
        expression = '5.0e-6*length_scale' 
    []

    [Second_Ring_Beam_Radius]
        type = ParsedMaterial
        property_name = rR2
        material_property_names = 'length_scale'
        expression = '49.0e-6*length_scale' 
    []

    [Second_ring_Beam_half_Thickness]
        type = ParsedMaterial
        property_name = rT2
        material_property_names = 'length_scale'
        expression = '1.0e-6*length_scale' 
    []

    [power]
        type = ParsedMaterial
        property_name = pow
        material_property_names = 'energy_scale time_scale'
        expression = '250*energy_scale/time_scale'
    []

    [volumetric_heat]
        type = BesselHS
        power = pow
        efficiency = 0.6
        a0 = 0.5 # gaussian_power_prop
        a1 = 0.3 # first_ring_power_prop
        a2 = 0.0 #${fparse 1-a0-a1} # second_ring_power_prop
        Ca = 2 # Coefficient Constant Outside Exponential
        Cb = 2 # Coefficient Constant Inside Exponential
        rG = rG
        rR1 = rR1
        rT1 = rT1
        rR2 = rR2
        rT2 = rT2               
        factor = 1.0e-4
        SGOrder_K = 1
        alpha = absorptivity
        function_x= path_x
        function_y= path_y
    []   

    [F_LIQUID]
        type = DerivativeParsedMaterial
        property_name = F1
        material_property_names = 'length_scale energy_scale v_mol'
        coupled_variables = 'temp'

        ## Polynomial Fitting
        # constant_names = 'factor_f1  a1   b1   c1   d1   e1   f1   g_1  h_1  x1'
        # constant_expressions = '12.3465  1337   18905   11.2    6.3   0.0023   0.0345    0.000019 -17.4  2227 '
        # expression = 'factor_f1*(a1-b1-c1*(temp-x1)+d1*(temp-x1)*log(e1*temp)+f1*(temp-x1)^2+g_1*(temp-x1)^3+h_1/temp)*energy_scale/(v_mol*length_scale^3)'    
            
        ## TDB Expression
        expression = '(24.9435*temp*log(1-exp(-126.68742/temp))-8.3145*temp*log(1+exp(-0.120271814300319*(19700.0-14.917*temp)/temp))-0.00067*temp^2.0-326.386169615)*energy_scale/(v_mol*length_scale^3)'
    []

    [F_FCC]
        type = DerivativeParsedMaterial
        property_name = F2
        material_property_names = 'length_scale energy_scale v_mol'
        coupled_variables = 'temp'

        ## Polynomial Fitting
        # constant_names = 'factor_f2  a2   b2   c2   d2   e2   f2   g_2  h_2  x2'
        # constant_expressions = '9.9782  300   4985   -49.663  -12   0.235  -0.001  0.00002   -0.001  235 '
        # expression = 'factor_f2*(a2-b2-c2*(temp-x2)+d2*(temp-x2)*log(e2*temp)+f2*(temp-x2)^2+g_2*(temp-x2)^3+h_2/temp)*energy_scale/(v_mol*length_scale^3)'    
    
        ## TDB Expression
        expression = '(24.9435*temp*log(1-exp(-126.68742/temp))+1.0*if(temp<1337.33,-0.001281018525*temp^2.0-4.02278360455656e-7*temp^3.0,1.55127e+36*temp^(-11.0)-2.27748e+18*temp^(-5.0)-2.0566206*temp*log(temp)+9.807219*temp+3898.778)-6103.656619615)*energy_scale/(v_mol*length_scale^3)'
    []

    [h1]
        type = SwitchingFunctionMaterial
        eta = eta1
        h_order = SIMPLE
        function_name = h1
    []

    [g1]
        type = BarrierFunctionMaterial
        g_order = SIMPLE
        eta = eta1
        function_name = g1
    []


    [mu]
        type = ParsedMaterial
        property_name = mu
        material_property_names = 'sigma delta energy_scale length_scale'
        expression = '6*(sigma/delta)*(energy_scale/length_scale^3)'
    []

    [kappa]
        type = ParsedMaterial
        property_name = kappa
        material_property_names = 'sigma delta energy_scale length_scale'
        expression = '0.75*(sigma*delta)*(energy_scale/length_scale)'
    []

    # [Mobility]
    #     type = ParsedMaterial
    #     property_name = M
    #     material_property_names = 'length_scale time_scale energy_scale M1 M2 h1'
    #     expression = '(h1*M1 + (1-h1)*M2)*(length_scale^5/(time_scale*energy_scale))'
    # []

    [L1-2]
        type = ParsedMaterial
        property_name = L1_2
        constant_names = factor_L
        constant_expressions = '1.0'
        material_property_names = 'M1 M2 length_scale time_scale energy_scale mu kappa'
        expression = 'factor_L*(0.5/0.2)*(M1+M2)*(mu/kappa)*(length_scale^3/(time_scale*energy_scale))'
    []

    [Interface_Mobility]
        type = ParsedMaterial
        property_name = L
        coupled_variables = 'eta1'
        material_property_names = 'L1_2 h1'
        expression = 'L1_2*h1*(1-h1)'
    []

    [free_energy]
        type = DerivativeTwoPhaseMaterial
        property_name = F
        fa_name = F2
        fb_name = F1
        coupled_variables = 'temp'
        eta = eta1
        derivative_order = 2
        outputs = exodus
        output_properties = 'F dF/dc dF/deta d^2F/dc^2 d^2F/dcdeta d^2F/deta^2'
        h = h1
        g = g1
    []


[]    


[Kernels]

    [mass]
        type = INSMass
        variable = p
        u = vel_x
        v = vel_y
        pressure = p
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

     # Kernels for Allen-Cahn equation for eta1
    [deta1dt]
        type = TimeDerivative
        variable = eta1
    []

    [ACInterface1]
        type = ACInterface
        variable = eta1
        kappa_name = kappa
        mob_name = L
    []

    [ACBulk1]
        type = AllenCahn
        variable = eta1
        coupled_variables = 'temp'
        f_name = F
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

    end_time            = 25
    dt                  = 1.2e-1

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
    # Area of Phases
   [area_h1]
       type = ElementIntegralMaterialProperty
       mat_prop = h1
       execute_on = 'Initial TIMESTEP_END'
   []

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
