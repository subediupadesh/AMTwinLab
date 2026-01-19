import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
from scipy.io import loadmat
from matplotlib.colors import ListedColormap
import io

colormap_data = loadmat('cold2hot.mat')
cold2hot = ListedColormap(colormap_data['colormap'])
hot2cold = ListedColormap(colormap_data['colormap'][::-1])

st.set_page_config(layout="wide")
inline_css = """div[data-testid="stExpander"] div[role="button"] p {font-size: 3rem;}"""
st.markdown(f"<style>{inline_css}</style>", unsafe_allow_html=True)

url1 = 'https://link.springer.com/article/10.1007/s11837-023-06363-8' # Bessel, Ring
url2 = 'https://doi.org/10.1080/17445302.2014.937059' # Double Ellipsoide
url3 = 'https://doi.org/10.1080/17452759.2024.2308513' # Super Gaussian
url4 = 'https://doi.org/10.1016/j.matpr.2020.12.842' # Volumetric Gaussian

def __main__():

    st.title(f'[Gaussian Heat Source]({url3})')
    cm1, cm2 = st.columns([0.2,0.8])

    def plot_gaussian_heat_distribution(A, Ca, Cb, P, eta, r_G, factor, i):
        rG = r_G*1.0e-6 ## converting to meter
        A = A#*1e7

        cmaps = ['hot_r', 'balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']
        x = np.linspace(-100e-6, 100e-6, 500)
        y = np.linspace(-100e-6, 100e-6, 500)
        x, y = np.meshgrid(x, y)

        r = (x**2 + y**2)**0.5
        F = np.where(r_G>=r, 1, 0) * factor
        
        Q = F*((Ca*A*P*eta)/(np.pi*rG**2))*(np.exp(-Cb*(r**2/rG**2)))
        
        cm1.write('Colormap: '+cmaps[i])
        cm1.write(r'$Q_{peak} =$  '+f'{np.max(Q):.3e}'+r'  $W/m^2$')
        
        camera = dict(eye=dict(x=1.5, y=1.5, z=2.5))

        fig = go.Figure(data=[go.Surface(z=Q, x=x, y=y, cmin=0, cmax=Q.max(), colorscale=cmaps[i], opacity=1.0 )])
        fig.update_layout( scene_camera=camera, scene=dict( xaxis=dict( title='X-axis', showgrid=False), yaxis=dict(title='Y-axis',showgrid=False), zaxis=dict(title='Intensity', showgrid=False, range=[0, Q.max()])), width=1000, height=1000, plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
        fig.update_traces(colorbar=dict(title=r'Q W/m^2'), colorbar_title_font=dict(size=30, color='black'), colorbar_exponentformat='B', colorbar_nticks=6, colorbar_len=0.5, colorbar_borderwidth=0.0, colorbar_thickness=70, colorbar_orientation='v', colorbar_tickfont=dict(family='Sans', color='black', size=25))
        if st.button('Save Image'):
            fig.write_image("../../../2_Post_Processing/figures/RHS_3D.png", width=8000, height=8000)
            fig.write_image("../../../2_Post_Processing/figures/jpg/GHS_3D.jpg", width=8000, height=8000)
            st.success("Image saved inside: 2_Post_Processing/figures GHS_3D.png")

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
    power = cm1.slider(r'''Power $$(P) \, W $$''', min_value=1, max_value=500, value=250, step=1)
    eta = cm1.slider(r'''Efficiency $$(\eta)$$ ''', min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    beam_radius = cm1.slider(r'''Beam Radius $$(r_G$$ $$\mu m)$$''', min_value=10.0, max_value=100.0, value=70.0, step=0.1)
    A = cm1.slider(r'''Absorptivity $$(A \times 10^7/m)$$''', min_value=0.00001, max_value=20.0, value=1.0, step=0.1)
    Ca = cm1.slider(r'''Constant $$(C_a)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.1)
    Cb = cm1.slider(r'''Constant $$(C_b)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.1)
    i = cm1.slider('colormap', min_value=0, max_value=9, value=0, step=1)
    factor = 1 #1.0e-4

    fig, Q = plot_gaussian_heat_distribution(A, Ca, Cb, power, eta, beam_radius, factor, i)

    cm2.header(r'$Q =  \frac{C_aAP\eta}{\pi r_G^2 } \exp\left[-C_b(\frac{r^2}{r_G^2})\right]$')
    
    cm2.plotly_chart(fig, use_container_width=True)
    st.divider()

__main__()