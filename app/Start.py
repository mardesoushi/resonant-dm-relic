import streamlit as st

## Modulos python
import altair as alt
from regex import D

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import pairwise_distances
import tempfile
from haversine import Unit
import haversine as hs
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import os
import streamlit as st
import pandas as pd 
import numpy as np

### newmodules
import sympy
from sympy import I as Im
from sympy import conjugate
from sympy import sympify, symbols, simplify, Symbol, Mul, expand
from sympy.parsing.latex import parse_latex
import antlr4

st.set_page_config(
    page_title="LaTeX inputer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

###########################################################################################################################
# Code start
st.title('Partial Wave Calculator')


lagrangian = st.text_input("Insert your Lagrangian")

st.markdown('''Your inputed lagrangian is: ''')
st.latex(rf'{lagrangian}')


st.markdown('Given a Sommerfeld correcti as:')

from PIL import Image
from pix2tex.cli import LatexOCR

img = Image.open('path/to/image.png')
model = LatexOCR()
print(model(img))