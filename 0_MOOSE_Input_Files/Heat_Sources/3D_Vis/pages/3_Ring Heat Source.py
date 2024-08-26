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


def __main__():

    st.title(f'[Ring Heat Source]({url1})')
    cm1, cm2 = st.columns([0.2,0.8])
    ls, ts, es = 1e6, 1e0, 1e9


    def plot_ring_heat_distribution(P, eta, r_R, r_T, A, Ca, Cb, factor, i ):
        
        rR, rT = r_R*1.0e-6, r_T*1.0e-6  # scaling unit to micro meter
        rG = 70*1.0e-6
        cmaps = ['hot_r', 'balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']        
        
        x = np.linspace(-100e-6, 100e-6, 500)
        y = np.linspace(-100e-6, 100e-6, 500)
        x, y = np.meshgrid(x, y)
        A = A#*1e7/1e6

        r = (x**2 + y**2)**0.5
        F = np.where((rR-(rT/2)<=r) & (r<=rR+(rT/2)), 1, 0) * factor
        Y = np.exp(-rR**2/(2*(rT/2)**2)) + (rR/(rT/2))*(np.pi/2)**0.5 * math.erfc(-rR/((rT/2)*2**0.5))
        Qmax = ((Ca*A*P*eta)/(np.pi*rG**2))

        Q = F*((Ca*A*P*eta)/(np.pi*rT**2 * Y)) * (np.exp(-Cb*((r-rR)**2/(rT**2))))
        cm1.write('Cmap: '+cmaps[i])
        cm1.write(r'$Q_{peak} =$  '+f'{np.max(Q):.3e}'+r'  $W/m^2$')
        cm1.write(r'$Q_{Gpeak} =$  '+f'{np.max(Qmax):.3e}'+r'  $W/m^2$')        

        camera = dict(eye=dict(x=1.5, y=1.5, z=2.5))
        fig = go.Figure(data=[go.Surface(z=Q, x=x, y=y, cmin=0, cmax=Qmax, colorscale=cmaps[i], opacity=1.0 )])
        fig.update_layout( scene_camera=camera, scene=dict( xaxis=dict( title='X-axis', showgrid=False), yaxis=dict(title='Y-axis',showgrid=False), zaxis=dict(title='Intensity', showgrid=False,  range=[0, Qmax])), width=1000, height=1000, plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
        fig.update_traces(colorbar=dict(title=r'Q W/m^2'), colorbar_title_font=dict(size=30, color='black'), colorbar_exponentformat='B', colorbar_nticks=6, colorbar_len=0.5, colorbar_borderwidth=0.0, colorbar_thickness=70, colorbar_orientation='v', colorbar_tickfont=dict(family='Sans', color='black', size=25))
        if st.button('Save Image'):
            fig.write_image("../../../2_Post_Processing/figures/RHS_3D.png", width=8000, height=8000)
            fig.write_image("../../../2_Post_Processing/figures/jpg/RHS_3D.jpg", width=8000, height=8000)
            st.success("Image saved inside: 2_Post_Processing/figures RHS_3D.png")

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
    power = cm1.slider(r'''Power $$(P) \, W $$''', min_value=1.0, max_value=500.0, value=250.0, step=1.0)
    eta = cm1.slider(r'''Efficiency $$(\eta)$$ ''', min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    rR = cm1.slider(r'''Ring Radius $$(r_r$$ $$\mu m)$$''', min_value=10.0, max_value=100.0, value=70.0, step=0.1)
    rT = cm1.slider(r'''Ring Beam thickness $$(r_t$$ $$\mu m)$$''', min_value=1.0, max_value=100.0, value=20.0, step=0.1)
    A = cm1.slider(r'''Absorptivity $$(A \times 10^7/m)$$''', min_value=0.00001, max_value=20.0, value=1.0, step=0.1)
    Ca = cm1.slider(r'''Constant $$(C_a)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.1)
    Cb = cm1.slider(r'''Constant $$(C_b)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.1)
    i = cm1.slider('colormap', min_value=0, max_value=10, value=0, step=1)
    factor = 1 #1.0e-4

    fig, Q = plot_ring_heat_distribution(power, eta, rR, rT, A, Ca, Cb, factor, i)

    
    cm2.header(r'$Q =  \frac{C_aAP\eta}{\pi r_t^2 \text{Y}(r_r,r_t)} \exp\left[-C_b\left(\frac{(r-r_r)^2}{r_t^2}\right)\right]$')
    # cm2.write(r'$\text{Y}(r_r,r_t) =  \exp\left(\frac{-r_r^2}{2r_t^2}\right) +\frac{r_r}{r_t}\sqrt{\frac{\pi}{2}}\, \text{erfc}(\frac{-r_r}{\sqrt{2}r_t}) $')
    cm2.write(r'$\text{Y}(r_r,r_t) =  \exp\left(\frac{-2r_r^2}{r_t^2}\right) +\frac{2r_r}{r_t}\sqrt{\frac{\pi}{2}}\, \text{erfc}(\frac{-\sqrt{2}r_r}{r_t}) $')
    
    cm2.plotly_chart(fig, use_container_width=True)

    st.divider()

__main__()