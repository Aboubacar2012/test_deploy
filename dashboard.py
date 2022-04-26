import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler, StandardScaler
from lime import lime_tabular
from sklearn.linear_model import LogisticRegression
import lime
from lime import lime_tabular
import streamlit.components.v1 as components
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import dill
import plotly.graph_objects as go
import plotly.express as px

df = pd.read_csv("TEST_FINAL_SCALE_2.csv")
print(df.shape)
#Xclient = pd.read_csv("Base_Clients_scale.csv")

#Télécharger modele
model = pickle.load(open('Credit_model_reg.pkl','rb'))
my_dill = dill.load(open('Explainer.dat', "rb"))





# Téléchargerscaler
file_scaler= open('file_scale.dat','rb')
std_scale=pickle.load(file_scaler)
file_scaler.close()

#Télécharger le explainer
file_explain=open('Explainer.dat','rb')
explainer=dill.load(open('features.pkl','rb'))
file_explain.close()

#liste des identifiants
liste_id = df['SK_ID_CURR'].values


# Télécherger l'image
st.sidebar.image('image-gallery-01-11-2016-b.png')


html_temp = """ 
           <div style ="background-color:darkgreen;padding:13px"> 
           <h1 style ="color:black;text-align:center;"> Dashboard Scoring Credit
           </h1> 
           </div> 
           <p></p>
           <div class="txt">Cette application est un Dashboard pour le chargé d’étude client.</div>
           <p></p>
           """
    # display the front end aspect
st.markdown(html_temp, unsafe_allow_html=True)


#affichage formulaire
#st.title('Dashboard Scoring Credit')
st.markdown("Prédictions de scoring client, notre seuil de choix est de 34 %.")
hobby = st.selectbox(" Veuillez choisir un identifiant à saisir: ", liste_id)





#select_box=st.sidebar.multiselect(label='Veuillez choisir un identifiant à saisir ',options=liste_id)
#st.markdown("Essayez avec les identifiants clients suivants : 100001, 100005,100013, 100042, 100066, 100074, 10015, 100107")
id_input = st.number_input(label='Veuillez saisir l\'identifiant du client demandeur de crédit:',format="%i", value=0)
if id_input not in liste_id:

    st.write("L'identifiant client n'est pas bon")


elif (int(id_input) in liste_id):
    i = df['SK_ID_CURR']==id_input
    Y = df[i]
    Y = Y.drop(['SK_ID_CURR'], axis=1)
    #scaler = StandardScaler()
    #scaler.fit_transform(Y)


#transformation des données
    #num = np.array(scaler.fit_transform(Y))
    num=np.array(Y)

#afficher les données
    st.subheader('Les données du client')
    st.write(id_input)
    st.write(Y)


# importer le modele et l'appliquer
    pr = model.predict_proba(num)[:, 1]
    if pr > 0.5:

        prevision= 'Rejet de la demande de crédit'

    else :

        prevision= 'Acceptation demande de crédit'
    #affichage prévision
    st.subheader('Le statut de la demande de crédit')
    st.write(prevision)
    st.subheader('Le pourcentage de scoring du client')
    st.write((pr[0]*100).round(0))


    explainer1=lime_tabular.LimeTabularExplainer(
    training_data = num,
    feature_names=df.columns,
    class_names=['Crédit Accepté','Crédit Refusé'],
    mode='classification',
    discretize_continuous=False
    )


    #interprétabilité
    st.subheader('Interprétabilité de notre prévision')
    #exp = explainer.explain_instance(data_rows=num.reshape(1,Y.shape[1])[0],predict_fn=model.predict_proba)
    exp =  explainer1.explain_instance(data_row=num[0],predict_fn=model.predict_proba)
    #exp = explainer.LimeTabularExplainer(training_data=num.reshape(1,Y.shape[1])[0],predict_fn=model.predict_proba)
    exp.show_in_notebook(show_table=True)
    
    
    components.html(exp.as_html(), height=800)
    plt.tight_layout()
    exp.as_list()
    
    st.markdown("### Deuxième graphique")
    exp.show_in_notebook(show_table=True)
    fig = exp.as_pyplot_figure()
        #plt.tight_layout()
    st.write(fig)


    #graphique
    #tickers=Y.columns
    #print(tickers)
    st.sidebar.subheader('Exploration des caractéristiques du client')
    #dropdown=st.multiselect('Choisir une variable du client',tickers)
    select_box=st.sidebar.multiselect(label='Features',options=Y.columns)
    plt.figure()
    fig1 = px.bar(Y, x=select_box)
    st.plotly_chart(fig1)
    #fig1.show()
    st.sidebar.subheader("Exploration des caractéristiques de l'ensemble des clients")
    select_box1 = st.sidebar.multiselect(label='Features', options=df.columns)
    fig2 = px.histogram(df, x=select_box1,nbins=50)
    st.plotly_chart(fig2)
    #fig2.show()

    # graphique deux variables quantitatives
    st.sidebar.subheader("Analyse Bivariée des caractéristiques de nos clients")
    select_box2 = st.sidebar.selectbox(label='Axe des abscisses', options=df.columns)
    select_box3 = st.sidebar.selectbox(label='Axe des ordonnées', options=df.columns)
    fig3 = px.scatter(df, x=select_box2, y=select_box3)
    st.plotly_chart(fig3)
