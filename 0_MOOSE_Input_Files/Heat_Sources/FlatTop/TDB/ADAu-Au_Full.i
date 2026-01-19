#  mpiexec -n 10 tumbleweed-opt --n-threads=2 -i ADThermal.i
## This is the moving laser hear source simulation for 2 eta 
## Written on 1st of July 2025

boltzmann_constant  = '${fparse 5.67e-8 * energy_scale/(time_scale*(length_scale)^2)}' # Scaling J/sm^2K^4
T_inf               = 300   #K
length_scale        = 1.0e6 # micrometers
time_scale          = 1.0e0 # seconds
energy_scale        = 1.0e9 # nano Joules
dt                  = 0.06  # seconds
end_time            = 28.0  # seconds
v_mol               = 10.21e-6  ## Molar Volume of Gold
gravity             = '${fparse -9.81 * length_scale/time_scale^2}'


[Mesh]
    type        = GeneratedMesh
    dim         = 2
    nx          = 200
    ny          = 100
    xmin        = 0
    xmax        = 1000
    ymin        = 0
    ymax        = 250
    elem_type   = QUAD9
[]

[Variables]
    [temp]
        initial_condition = 300
    []

    [eta1]
        order   = FIRST
        family  = LAGRANGE
    []

    [vel]
        order   = SECOND
        family  = LAGRANGE_VEC
        outputs = 'exodus'
    []

    [p]
    []
[]


[ICs]
    [eta1]
        variable    = 'eta1'
        type        = FunctionIC
        function    = 'if(y>200&y<=250 & x>=100&x<=150, 1, 0)'
    []

    [velocity]
        type        = VectorFunctionIC
        variable    = 'vel'
        function_x  = 'if(y>200&y<=250 & x>=100&x<=150, 1e-2, 1e-6)'
        function_y  = 'if(y>200&y<=250 & x>=100&x<=150, 1e-2, 1e-6)'
    []
[]


[BCs]

    # 1D: left = 0, right = 1
    # 2D: bottom = 0, right = 1, top = 2, left = 3
    # 3D: back = 0, bottom = 1, right = 2, top = 3, left = 4, front = 5

    # [temp_fixed]
    #     type                        = ADDirichletBC
    #     variable                    = 'temp'
    #     boundary                    = 'bottom'
    #     value                       = 300
    # []

    [radiation_flux]
        type                        = FunctionRadiativeBC
        variable                    = 'temp'
        boundary                    = 'left top right'
        emissivity_function         = '(50*${energy_scale}/(${time_scale}*${length_scale}^2))/(${boltzmann_constant}*4*${T_inf}^3)'  ##https://mooseframework.inl.gov/source/bcs/FunctionRadiativeBC.html
        Tinfinity                   = '${T_inf}'
        stefan_boltzmann_constant   = '${boltzmann_constant}'
    []

    [convectiveFlux_air]
        type                        = ConvectiveHeatFluxBC
        variable                    = 'temp'
        boundary                    = 'left top right'
        T_infinity                  = '${T_inf}'
        heat_transfer_coefficient   = '${fparse 50* energy_scale/(time_scale*length_scale^2)}' # 50 W/m^2K for air  https://doi.org/10.1533/978-1-78242-164-1.353
        heat_transfer_coefficient_dT = 0
    []

    [convectiveFlux_metal]
        type                        = ConvectiveHeatFluxBC
        variable                    = 'temp'
        boundary                    = 'bottom'
        T_infinity                  = '${T_inf}'
        heat_transfer_coefficient   = '${fparse 11500* energy_scale/(time_scale*length_scale^2)}' # 11500 W/m^2K for metal  https://doi.org/10.1016/j.intermet.2017.11.021  # 1.15e6
        heat_transfer_coefficient_dT = 0
    []

    [neumann1]
        type                        = NeumannBC
        boundary                    = 'bottom'
        variable                    = 'eta1'
        value                       = 0
    []

    [no_slip]
        type                        = ADVectorFunctionDirichletBC
        variable                    = 'vel'
        boundary                    = 'bottom right left'
        function_x                  = '0'
        function_y                  = '0'
    []  

    [surface_tension]
        type                        = INSADSurfaceTensionBC
        variable                    = 'vel'
        boundary                    = 'top'
        include_gradient_terms      = true      ## Surface tension of Gold https://www.sciencedirect.com/science/article/pii/S0167577X1100440X
    []

    [vapor_recoil]
        type                        = INSADVaporRecoilPressureMomentumFluxBC
        variable                    = 'vel'
        boundary                    = 'top'
        rc_pressure_name            = 'rc_pressure'
    []    
[]


[Functions]
    [path_x]
        type        = ParsedFunction
        expression  = 125+30.0*t
    []

    [path_y]
        type        = ParsedFunction
        expression  = 250
    []

    [laser_switch]
        type        = ParsedFunction
        expression  = 'if(t<=25, 1, 0)'
    []
[]


[Materials]
   ## Heat-Conduction Materials 
    [conductivity_values]
        type                    = ADGenericConstantMaterial
        prop_names              = 'pseudo_k1   pseudo_k2'
        prop_values             = '100         338.91'      # https://doi.org/10.1016/j.jestch.2023.101413
    []

    [conductivity_LIQUID]
        type                    = ADDerivativeParsedMaterial
        property_name           = 'k1'
        material_property_names = 'pseudo_k1'
        constant_names          = 'f_k1    alpha_k1'
        constant_expressions    =  '1      0.027397'                  # Value of alpha_k3 = 0.027397 W/mK^2 reference: https://github.com/anilkunwar/temperature_dependent_material_properties/blob/main/getAuThermalConductivity.F90
        coupled_variables       = 'temp'
        expression              = 'f_k1*(pseudo_k1 + alpha_k1*temp)'
    []

    [conductivity_FCC]
        type                    = ADDerivativeParsedMaterial
        property_name           = 'k2'
        material_property_names = 'pseudo_k2'
        constant_names          = 'f_k2     alpha_k2'
        constant_expressions    = '1        -6.93e-2'                      # Value of alpha_k4 = -6.93e-2 W/mK reference: https://github.com/anilkunwar/temperature_dependent_material_properties/blob/main/getAuThermalConductivity.F90
        coupled_variables       = 'temp'
        expression              = 'f_k2*(pseudo_k2 + alpha_k2*temp)'
    []

    [conductivity]
        type                    = ADParsedMaterial
        property_name           = 'thermal_conductivity'
        material_property_names = 'k1 k2 h1'
        expression              = '(h1*k1 + (1-h1)*k2) * ${energy_scale}/(${length_scale}*${time_scale})'
    []

    [density_values]
        type                    = ADGenericConstantMaterial
        prop_names              = 'pseudo_rho1     pseudo_rho2'
        prop_values             = '19325.28        19657.6'             # https://doi.org/10.1016/j.jestch.2023.101413
    []

    [density_LIQUID]
        type                    = ADDerivativeParsedMaterial
        property_name           = 'rho1'
        material_property_names = 'pseudo_rho1'
        constant_names          = 'f_density1   alpha_rho1'
        constant_expressions    = '1              -1.44'                       # Value of alpha_rho3 = -1.44 reference: https://github.com/anilkunwar/temperature_dependent_material_properties/blob/main/getAuDensity.F90
        coupled_variables       = 'temp'
        expression              = 'f_density1*(pseudo_rho1 + alpha_rho1*temp)'
    []

    [density_FCC]
        type                    = ADDerivativeParsedMaterial
        property_name           = 'rho2'
        material_property_names = 'pseudo_rho2'
        constant_names          = 'fos_density2 alpha_rho2'
        constant_expressions    = '1 -1.2'                                     # Value of alpha_rho4 = -1.2 reference: https://github.com/anilkunwar/temperature_dependent_material_properties/blob/main/getAuDensity.F90
        coupled_variables       = 'temp'
        expression              = 'fos_density2*(pseudo_rho2 + alpha_rho2*temp)'
    []

    [density]
        type                    = ADParsedMaterial
        property_name           = 'density_name'
        material_property_names = 'rho1 rho2 h1'
        expression              = '(h1*rho1 + (1-h1)*rho2)/(${length_scale}^3)'
    []

    [spec_heat_values]
        type                    = ADGenericConstantMaterial
        prop_names              = 'pseudo_sp1  pseudo_sp2'
        prop_values             = '158         132'                           # https://webbook.nist.gov/cgi/inchi?ID=C7440326&Mask=2&Type=JANAFS&Table=on (convert J/molK to J/kgK) &  https://doi.org/10.1016/S0364-5916(01)00026-8
    []

    [Sp_LIQUID]
        type                    = ADDerivativeParsedMaterial
        property_name           = 'sp1'
        material_property_names = 'pseudo_sp1'
        constant_names          = 'f_sp1   alpha_sp1   beta_sp1'
        constant_expressions    = '1       5.08e7      -0.0114'                           # Fitted the data from: .TDB file from https://doi.org/10.1016/S0364-5916(01)00026-8
        coupled_variables       = 'temp'
        expression              = 'f_sp1*(alpha_sp1*exp(beta_sp1*temp)+pseudo_sp1)'
    []

    [Sp_FCC]
        type                    = ADDerivativeParsedMaterial
        property_name           = 'sp2'
        material_property_names = 'pseudo_sp2'
        constant_names          = 'f_sp2  alpha_sp2   beta_sp2'
        constant_expressions    = '1      2.5e-5      -0.011'                            # Fitted the data from: .TDB file from https://doi.org/10.1016/S0364-5916(01)00026-8
        coupled_variables       = 'temp'
        expression              = 'f_sp2*(alpha_sp2*temp^2 + beta_sp2*temp + pseudo_sp2)'
    []

    [specific_heat]
        type                    = ADParsedMaterial
        property_name           = 'specific_heat'
        material_property_names = 'sp1 sp2 h1'
        expression              = '(h1*sp1 + (1-h1)*sp2)*${energy_scale}'
    []

    [absorptivity_value]
        type                    = ADParsedMaterial
        property_name           = 'absorptivity'
        expression              = '8.5e7/${length_scale}'
    []

    [Gaussian_Beam_Radius]
        type                    = ADParsedMaterial
        property_name           = 'rG'
        expression              = '70.0e-6*${length_scale}'
    []

    [laser_power]
        type                    = ADParsedMaterial
        property_name           = 'P'
        expression              = '0.025*${energy_scale}/${time_scale}'
    []

    [volumetric_heat]
        type                    = ADLaserSource
        beam_type               = 'FlatTop'
        power                   = 'P'
        efficiency              = 0.75
        Ca                      = 2.0
        Cb                      = 2.0
        rG                      = 'rG'
        SGOrder_K               = 4.2
        alpha                   = 'absorptivity'
        function_x              = 'path_x'
        function_y              = 'path_y'
        laser_switch            = 'laser_switch' # Laser switch 1 for ON and 0 for OFF
    []

   ## Phase Field Materials 
    [pf_constants]
        type                    = ADGenericConstantMaterial
        prop_names              = 'sigma            delta                gamma       R      TAU'     # sigma -> J/m^2; delta -> meter;
        prop_values             = '0.5     ${fparse 10/length_scale}     1.5        8.31    0.2'
    []

    [F_LIQUID]
        type                    = ADDerivativeParsedMaterial
        property_name           = 'F1'
        coupled_variables       = 'temp'

        ## Polynomial Fitting
        # constant_names          = 'factor_f1  a1     b1      c1     d1      e1        f1        g_1     h_1    x1'
        # constant_expressions    = '10.412    1337   24474   16.2   24.5  0.00214   0.04512   0.000019  -17.4  2662'
        # expression              = 'factor_f1*(a1-b1-c1*(temp-x1)+d1*(temp-x1)*log(e1*temp)+f1*(temp-x1)^2+g_1*(temp-x1)^3+h_1/temp) * ${energy_scale}/(${v_mol}*${length_scale}^3)'
            
        ## TDB Expression
        expression              = '(24.9435*temp*log(1-exp(-126.68742/temp))-8.3145*temp*log(1+exp(-0.120271814300319*(19700.0-14.917*temp)/temp))-0.00067*temp^2.0-326.386169615) * ${energy_scale}/(${v_mol}*${length_scale}^3)'
    []

    [F_FCC]
        type                    = ADDerivativeParsedMaterial
        property_name           = 'F2'
        coupled_variables       = 'temp'

        ## Polynomial Fitting
        # constant_names          = 'factor_f2    a2     b2       c2      d2      e2       f2        g_2        h_2     x2'
        # constant_expressions    = '9.03         300    515    -46.2   -13.5   0.1985   -0.001    0.00002    -0.001   121'
        # expression              = 'factor_f2*(a2-b2-c2*(temp-x2)+d2*(temp-x2)*log(e2*temp)+f2*(temp-x2)^2+g_2*(temp-x2)^3+h_2/temp) * ${energy_scale}/(${v_mol}*${length_scale}^3)'
    
        ## TDB Expression
        expression              = '(24.9435*temp*log(1-exp(-126.68742/temp))+1.0*if(temp<1337.33,-0.001281018525*temp^2.0-4.02278360455656e-7*temp^3.0,1.55127e+36*temp^(-11.0)-2.27748e+18*temp^(-5.0)-2.0566206*temp*log(temp)+9.807219*temp+3898.778)-6103.656619615) * ${energy_scale}/(${v_mol}*${length_scale}^3)'
    []

    [h1]
        type                    = ADDerivativeParsedMaterial
        property_name           = 'h1'
        # expression              = '3*(if(eta1 > 1, 1, if(eta1 < 0, 0, eta1)))^2 - 2*(if(eta1 > 1, 1, if(eta1 < 0, 0, eta1)))^3'
        expression              = 'eta1^2 / (eta1^2 + (1 - eta1)^2)'
        coupled_variables       = 'eta1'
        derivative_order        = 2
        outputs                 = 'exodus'
    []

    [g1]
        type                    = BarrierFunctionMaterial
        g_order                 = 'SIMPLE'
        eta                     = 'eta1'
        function_name           = 'g1'
    []

    [g1_ADConverter]
        type                    = MaterialADConverter
        reg_props_in            = 'g1'
        ad_props_out            = 'g1_AD'
    []

    [mu]
        type                    = ADParsedMaterial
        property_name           = 'mu'
        material_property_names = 'sigma delta'
        expression              = '6*(sigma/delta)*(${energy_scale}/${length_scale}^3)'
    []

    [kappa]
        type                    = ADParsedMaterial
        property_name           = 'kappa'
        material_property_names = 'sigma delta'
        expression              = '3/4*(sigma*delta)*(${energy_scale}/${length_scale})'
    []

    [M1] # M_si unit-> m^5/Js
        type                    = ADParsedMaterial
        property_name           = 'M1'
        material_property_names = 'R'
        constant_names          = 'F_M1       M01       Q1'
        # constant_expressions    = '1e-19     3.15     1073.15'
        constant_expressions    = '1e-15       2.90     3043'
        coupled_variables       = 'temp'
        expression              = 'F_M1*M01*exp(-Q1/(R*temp)) * (${length_scale}^5/(${time_scale}*${energy_scale}))'
        outputs                 = 'exodus'
    []

    [M2]
        type                    = ADParsedMaterial
        property_name           = 'M2'
        material_property_names = 'R'
        constant_names          = 'F_M2       M02       Q2'
        # constant_expressions    = '1e-25     3.38     1165.84'
        constant_expressions    = '1e-21      4.88      2221'
        coupled_variables       = 'temp'
        expression              = 'F_M2*M02*exp(-Q2/(R*temp)) * (${length_scale}^5/(${time_scale}*${energy_scale}))'
        outputs                 = 'exodus'
    []

    [Mobility] ## CH-Mobility Not used in Allen-Cahn Equation
        type                    = ADParsedMaterial
        property_name           = 'M'
        material_property_names = 'M1 M2 h1'
        expression              = '(h1*M1 + (1-h1)*M2)'
    []

    [L1-2]
        type                    = ADParsedMaterial
        property_name           = 'L1_2'
        constant_names          = 'factor_L'
        constant_expressions    = '1.0'
        material_property_names = 'M1 M2 TAU mu kappa delta'
        expression              = 'factor_L*(4/3)*(mu/kappa)*((M1+M2)/(2*TAU))'
        outputs                 = 'exodus'
    []

    [Interface_Mobility]
        type                    = ADDerivativeParsedMaterial
        property_name           = 'L'
        coupled_variables       = 'eta1 temp'
        material_property_names = 'L1_2 h1'
        expression              = 'L1_2*h1*(1-h1)'
        outputs                 = 'exodus'
    []

    [free_energy]
        type                    = ADDerivativeParsedMaterial
        property_name           = 'F'
        material_property_names = 'F1(temp) F2(temp) h1(eta1) g1_AD(eta1)'
        coupled_variables       = 'eta1 temp'
        constant_names          = 'W'
        constant_expressions    = '0.0'
        expression              = 'h1*F1 + (1-h1)*F2 + W*g1_AD'
        derivative_order        = 2
        outputs                 = 'exodus'
    []

   ## Fluid Flow Materials
    [fluid_constants]
        type                    = ADGenericConstantMaterial
        prop_names              = 'PHI_Area     k_Btz        Es      Vs' # Vs = (Atomic wt in kg/mol)/(density at melting)
        prop_values             = '7.715e-20  1.38e-23     368000  1.132e-5'
    []

    # [recoil_pressure]
        # type                    = ADParsedMaterial
        # property_name           = 'rc_pressure'
        # coupled_variables       = 'temp'
        # expression              = 'if(temp<2807, 0, 0.56 * 101325 * exp(40144 * (1/2807 - 1/temp)) /(${length_scale}*${time_scale}^2))'
        # outputs                 = 'exodus'
    # []

    [surface_tension]  # https://doi.org/10.1016/j.matlet.2011.04.063
        type                    = ADDerivativeParsedMaterial
        property_name           = 'surface_tension'
        coupled_variables       = 'temp'
        material_property_names = 'PHI_Area k_Btz R Es Vs rho1'
        expression              = '((1/PHI_Area)*(Vs/(0.196967/rho1))^2 *k_Btz*temp * (0.139*(Es/(R*temp)) - 0.053)) * (${energy_scale}/${length_scale}^2)'
        outputs                 = 'exodus'
    []

    [marangoni_curvature]
        type                    = SurfaceTensionMaterial
        boundary                = 'top'
        coupled_variables       = 'temp'
        surface_tension_name    = 'surface_tension'
        outputs                 = 'exodus'
    []

    [mu_values]
        type                    = ADGenericConstantMaterial
        prop_names              = 'pseudo_mu1     mu2'
        prop_values             = '0.006        3.11e2'
    []

    [mu_LIQ]
        type                    = ADParsedMaterial
        property_name           = 'mu1'
        material_property_names = 'pseudo_mu1'
        constant_names          = 'Q_mu'
        constant_expressions    = '2669'
        coupled_variables       = 'temp'
        expression              = 'exp(-0.1990 + (Q_mu/temp)) / 1000' # The unit is in milli-Pascal-sec so divide by 1000 to convert to Pa.s ## https://link.springer.com/article/10.1007/s10765-016-2104-7
    []

    [mu_NS]
        type                    = ADParsedMaterial
        property_name           = 'mu_name'  # To distinguish with phase field mu
        material_property_names = 'mu1 mu2 h1'
        expression              = '(h1*mu1 + 10*mu1*h1*(1-h1) + (1-h1)*mu2) / (${length_scale}*${time_scale})'
    []

    [ins_mat]
        type                    = INSADStabilized3Eqn
        velocity                = 'vel'
        pressure                = 'p'
        temperature             = 'temp'
        rho_name                = 'density_name'
        mu_name                 = 'mu_name'
        k_name                  = 'thermal_conductivity'
        cp_name                 = 'specific_heat'
        outputs                 = 'exodus'
    []
[]    


[Kernels]
    [time]
        type                    = INSADHeatConductionTimeDerivative
        variable                = 'temp'
        # density_name            = 'density_name'
        # specific_heat           = 'specific_heat'
    []

    [heat_conduct]
        type                    = ADHeatConduction
        variable                = 'temp'
        thermal_conductivity    = 'thermal_conductivity'
    []

    [heat_source]
        type                    = ADMatHeatSource
        material_property       = 'volumetric_heat'
        variable                = 'temp'
    []


    [temperature_advection]
        type                    = INSADEnergyAdvection
        variable                = 'temp'
    []

    [temperature_supg]
        type                    = INSADEnergySUPG
        variable                = 'temp'
        velocity                = 'vel'
    []


    [deta1dt]
        type                    = ADTimeDerivative
        variable                = 'eta1'
    []

    [ACInterface1]
        type                    = ADACInterface
        variable                = 'eta1'
        kappa_name              = 'kappa'
        mob_name                = 'L'
        coupled_variables       = 'temp'
    []

    [ACBulk1]
        type                    = ADAllenCahn
        variable                = 'eta1'
        f_name                  = 'F'
        mob_name                = 'L'
    []


    [mass]
        type                    = INSADMass
        variable                = 'p'
    []

    [mass_pspg]
        type                    = INSADMassPSPG
        variable                = 'p'
        rho_name                = 'density_name'
    []

    [momentum_time]
        type                    = INSADMomentumTimeDerivative
        variable                = 'vel'
    []

    [momentum_viscous]
        type                    = INSADMomentumViscous
        variable                = 'vel'
        mu_name                 = 'mu_name'
        viscous_form            = 'laplace'
    []

    [momentum_pressure]
        type                    = INSADMomentumPressure
        variable                = 'vel'
        pressure                = 'p'
    []

    [momentum_supg]
        type                    = INSADMomentumSUPG
        variable                = 'vel'
        velocity                = 'vel'
    []

    [fluid_momentum_gravity]
        type                    = INSADGravityForce
        variable                = 'vel'
        gravity                 = '0 ${gravity} 0'
    []

    [fluid_momentum_buoyancy]
        type                    = INSADBoussinesqBodyForce
        variable                = 'vel'
        gravity                 = '0 ${gravity} 0'
        alpha_name              = '42.0e-6' ## Thermal Expansion Coefficient of LIQ Au (/K)
        ref_temp                = '1337'
        temperature             = 'temp'
    []

    [momentum_advection]
        type                    = INSADMomentumAdvection
        variable                = 'vel'
    []
[]


[Executioner]
    type                    = Transient
    solve_type              = 'PJFNK'
    automatic_scaling       = true
    
    petsc_options           = '-snes_converged_reason -ksp_converged_reason -options_left'
    petsc_options_iname     = '-ksp_gmres_restart -pc_factor_shift_type -pc_factor_shift_amount -pc_type'
    petsc_options_value     = '100 NONZERO 1e-15 ilu'

    # petsc_options_iname     = '-pc_type -pc_hypre_type -ksp_gmres_restart -pc_hypre_boomeramg_strong_threshold'
    # petsc_options_value     = 'hypre    boomeramg      31       0.7'

    l_max_its               = 30
    nl_max_its              = 50
    l_tol                   = 1e-04
    nl_rel_tol              = 1e-08
    nl_abs_tol              = 1e-09

    end_time                = '${end_time}'
    dt                      = '${dt}'

    # [Adaptivity]
    #     initial_adaptivity  = 1
    #     refine_fraction     = 0.7
    #     coarsen_fraction    = 0.1
    #     max_h_level         = 2
    #     # weight_names        = 'eta1'
    #     # weight_values       = '1'
    # []

[]


[Preconditioning]
    active                  = 'SMP'
    
    [SMP]
        type                = SMP
        full                = true
        petsc_options_iname = '-pc_type -pc_factor_shift_type -pc_factor_mat_solver_type'
        petsc_options_value = 'lu       NONZERO               strumpack'
    []

    [mydebug]
        type                = FDP
        full                = true
    []
[]


[Postprocessors]
    [area_h1]
        type                 = ADElementIntegralMaterialProperty
        mat_prop             = 'h1'
        execute_on           = 'TIMESTEP_END'
    []

    [temp_max]
        type                = ElementExtremeValue
        variable            = 'temp'
        value_type          = 'max'
    []

    [temp_avg]
        type                = ElementAverageValue
        variable            = 'temp'
    []

    [temp_min]
        type                = ElementExtremeValue
        variable            = 'temp'
        value_type          = 'min'
    []

    [dt]
        type                = TimestepSize
    []
[]

[Outputs]
   exodus                   = true
   time_step_interval       = 1
   file_base                = 'exodus/ThermoPhaseFluid/TPF'
   csv = true
   [my_checkpoint]
       type                 = Checkpoint
       num_files            = 2
       time_step_interval   = 2
       file_base            = 'exodus/ThermoPhaseFluid/TPF'
   []
[]

[Debug]
    show_var_residual_norms = true
[]
