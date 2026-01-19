import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import io

st.set_page_config(layout="wide")
inline_css = """div[data-testid="stExpander"] div[role="button"] p {font-size: 3rem;}"""
st.markdown(f"<style>{inline_css}</style>", unsafe_allow_html=True)

url1 = 'https://link.springer.com/article/10.1007/s11837-023-06363-8' # Bessel, Ring
url2 = 'https://doi.org/10.1080/17445302.2014.937059' # Double Ellipsoide
url3 = 'https://doi.org/10.1080/17452759.2024.2308513' # Super Gaussian
url4 = 'https://doi.org/10.1016/j.matpr.2020.12.842' # Volumetric Gaussian
url5 = 'https://www.sciencedirect.com/science/article/pii/S0079672714000317' # Bessel with multiple ring

def __main__():

    st.title(f'[Bessel]({url1}) [ Heat Source]({url5})')
    cm1, cm2 = st.columns([0.2,0.8])

    def plot_bessel_heat_distribution(a0, a1, P, eta, r_g, rr_1, r_t1, rr_2, r_t2, A, Ca, Cb, factor, i ):
        rBG, rR1, rT1, rR2, rT2, = r_g*1.0e-6, rr_1*1.0e-6, r_t1*1.0e-6, rr_2*1.0e-6, r_t2*1.0e-6,  # scaling unit to meter
        A = A#*1e7/1e6
        rG = 70*1.0e-6
        a2 = 1-a0-a1
        # a2 = 0
        cmaps = ['hot_r', 'balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']
        x = np.linspace(-100e-6, 100e-6, 500)
        y = np.linspace(-100e-6, 100e-6, 500)
        x, y = np.meshgrid(x, y)
        r = (x**2 + y**2)**0.5

        RF1   = 3.0e-6
        F0    = np.where(rBG>r, 1, 0) * factor
        F1    = np.where((rR1-(rT1/2)+RF1<=r) & (r<=rR1+(rT1/2)-RF1), 1, 0) * factor
        F2    = np.where((rR2-(rT2/2)<=r) & (r<=rR2+(rT2/2)), 1, 0) * factor

        Y1 = np.exp(-rR1**2/(2*(rT1/2)**2)) + (rR1/(rT1/2))*(np.pi/2)**0.5 * math.erfc(-rR1/((rT1/2)*2**0.5))
        Y2 = np.exp(-rR2**2/(2*(rT2/2)**2)) + (rR2/(rT2/2))*(np.pi/2)**0.5 * math.erfc(-rR2/((rT2/2)*2**0.5))
     
        Qmax = ((Ca*A*P*eta)/(np.pi*rG**2))
        Q0 = F0*((Ca*eta*A*a0*P)/(np.pi*rBG**2))*(np.exp(-Cb*(r**2/rBG**2)))
        Q1 = F1*((Ca*eta*A*a1*P)/(np.pi*rT1**2*Y1)) * (np.exp(-Cb*((r-rR1)**2/(rT1**2))))
        Q2 = F2*((Ca*eta*A*a2*P)/(np.pi*rT2**2*Y2)) * (np.exp(-Cb*((r-rR2)**2/(rT2**2))))
        Q = Q0 + Q1 + Q2

        cm1.write('Cmap: '+cmaps[i])
        cm1.write(r'$a_0 + a_1 + a_2 + a_3 = 1$')
        cm1.write(r'Q$_G^{peak}$: '+f'{np.max(Q0):.3e}'+r'  $W/m^2$')
        cm1.write(r'Q$_{R1}^{peak}$: '+f'{np.max(Q1):.3e}'+r'  $W/m^2$')
        cm1.write(r'Q$_{R2}^{peak}$: '+f'{np.max(Q2):.3e}'+r'  $W/m^2$')
        cm1.write(r'$Q_{Gpeak} =$  '+f'{np.max(Qmax):.3e}'+r'  $W/m^2$')

        camera = dict(eye=dict(x=1.5, y=1.5, z=2.5))
        fig = go.Figure(data=[go.Surface(z=Q, x=x, y=y, cmin=0, cmax=Qmax, colorscale=cmaps[i], opacity=1.0 )])
        fig.update_layout( scene_camera=camera, scene=dict( xaxis=dict( title='X-axis', showgrid=False), yaxis=dict(title='Y-axis',showgrid=False), zaxis=dict(title='Intensity', showgrid=False,  range=[0, Qmax])), width=1000, height=1000, plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
        fig.update_traces(colorbar=dict(title=r'Q W/m^2'), colorbar_title_font=dict(size=30, color='black'), colorbar_exponentformat='B', colorbar_nticks=6, colorbar_len=0.5, colorbar_borderwidth=0.0, colorbar_thickness=70, colorbar_orientation='v', colorbar_tickfont=dict(family='Sans', color='black', size=25))
        if st.button('Save Image'):
            fig.write_image("../../../2_Post_Processing/figures/BHS_3D.png", width=8000, height=8000)
            fig.write_image("../../../2_Post_Processing/figures/jpg/BHS_3D.jpg", width=8000, height=8000)
            st.success("Image saved inside: 2_Post_Processing/figures BHS_3D.png")

        # if st.button('Save Image as PNG'):
        #     with io.BytesIO() as buffer:
        #         fig.write_image(buffer, format='png')
        #         buffer.seek(0)
        #         st.download_button(
        #             label="Download PNG",
        #             data=buffer,
        #             file_name="HS_3D.png",
        #             mime="image/png"
        #         )
        return fig, Q


    cm1.header('Parameters')

    P = cm1.slider(r'Power [P] \, W ',                                      min_value=1.0,  max_value=1000.0,   value=250.0,    step=1.0)
    eta = cm1.slider(r'Efficiency $[\eta]$',                                min_value=0.0,  max_value=1.0,      value=0.75,      step=0.0001)
    rG = cm1.slider(r'''Gaussian Beam Radius $$(r_G$$ $$\mu m)$$''',        min_value=1.0,  max_value=75.0,     value=22.65,     step=0.1)
    r1 = cm1.slider(r'''Ring 1 radius $$(Rr_1$$ $$\mu m)$$''',              min_value=1.0,  max_value=100.0,    value=40.0,     step=0.1)
    rT1 = cm1.slider(r'''Ring 1 half thickness $$(r_{t1}$$ $$\mu m)$$''',   min_value=1.0,  max_value=50.0,     value=30.0,      step=0.01)
    r2 = cm1.slider(r'''Ring 2 radius $$(Rr_2$$ $$\mu m)$$''',              min_value=1.0,  max_value=100.0,    value=70.0,     step=0.1)
    rT2 = cm1.slider(r'''Ring 2 half thickness $$(r_{t2}$$ $$\mu m)$$''',   min_value=1.0,  max_value=25.0,     value=20.0,      step=0.01)
    

    a0 = cm1.slider(r'P$_{G} \, proportion \, \, (a_0)$',    min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    a1 = cm1.slider(r'P$_{R1} \, proportion \, \, (a_1)$', min_value=0.0, max_value=a0,  value=0.5, step=0.01)
    
    A = cm1.slider(r'''Absorptivity $$(A \times 10^7/m)$$''', min_value=0.00001, max_value=20.0, value=1.00, step=0.1)
    C1 = cm1.slider(r'''Constant $$(C_1)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.1)
    C2 = cm1.slider(r'''Constant $$(C_2)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.1)
    i = cm1.slider('Cmap  ', min_value=0, max_value=9, value=0, step=1)
    factor = 1#1.0e-4

    fig, Q = plot_bessel_heat_distribution(a0, a1, P, eta, rG, r1, rT1, r2, rT2, A, C1, C2, factor, i, )

    cm2.subheader(r'$Q = Q_G + \displaystyle \sum _{i=1}^{n} Q_{R_i}$')
    cm3, cm4= cm2.columns([0.5, 0.5])
    cm3.subheader(r'$Q_G = \frac{A^{1/k}k}{\Gamma(1/k)} \frac{a_0P\eta C1}{\pi r_G^2} \exp\left[-C_2\left(\frac{r^2}{r_G^2}\right)^k\right]$')
    cm4.subheader(r'$Q_{R_n} =  \frac{a_nP\eta C_1}{\pi r_{t_n}^2 \text{Y}(r_{r_n},r_{t_n})} \exp\left[-C_2\left(\frac{(r-r_{r_n})^2}{r_{t_n}^2}\right)\right]$')
    # cm4.write(r'$a_0 + a_1 + a_2 + a_3 = 1$')

    
    cm2.plotly_chart(fig, use_container_width=True)
    st.divider()

__main__()