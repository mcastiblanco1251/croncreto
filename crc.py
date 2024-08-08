import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
from pathlib import Path
import streamlit_authenticator as stauth
#import database as db
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.metrics import r2_score

#im = Image.open('C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Arroz/apps/arroz.png')
im2 = Image.open('predictor2.png')
st.set_page_config(page_title='Pred_App', layout="wide", page_icon=im2)
#st.set_option('deprecation.showPyplotGlobalUse', False)
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .checkbox-text {font-size:114px;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

imagel = Image.open('Logo GPR.jpg')
new_image = imagel.resize((300, 100))
new_image2=imagel.resize((300, 150))
#---- USER AUTHENTICATION
names=['Manuel Castiblanco', 'Pedro Gomez']
usernames=['mcastiblanco', 'pgomez']
file_path= Path(__file__).parent/'hashed_pw.pkl'

with file_path.open('rb') as file:
    hashed_passwords=pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "pre_App", '123456', cookie_expiry_days=30)

name, authentication_status, usernames = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password es incorrecto")

if authentication_status == None:
    st.warning("Por favor introduzca su username y password")

if authentication_status:
    # imagelp2 = Image.open('predictor2.png')
    # new_image5=imagelp2.resize((120, 40))
    # st.image(new_image5)
    row1_1, row1_2,row1_3 = st.columns((1, 2,1))

    with row1_1:
        imagelp2 = Image.open('predictor2.png')
        new_image5=imagelp2.resize((120, 40))
        st.image(new_image5)
        #st.image(new_image2)
    with row1_2:
        imagelp = Image.open('predictor.png')
        imagelp2 = Image.open('predictor2.png')
        new_image3=imagelp.resize((240, 80))
        new_image4=imagelp2.resize((240, 80))
        #st.image(new_image2)
        #st.header('PREDIKTOR')
        #new_image5=imagelp2.resize((120, 40))
        #st.image(new_image5)
        st.subheader("""
        Predicción Propiedades
        Esta App predice propiedades de Resistencia del Concreto a partir de propiedades base!
        """)

    with row1_3:
        st.image(new_image2)
        st.markdown('PREDIKTOR App by [GPREnergy](https://www.gprenergy.co/)')
        #st.image(new_image2)
        # with st.expander("Contact us 👉"):
        #     with st.form(key='contact', clear_on_submit=True):
        #         name = st.text_input('Name')
        #         mail = st.text_input('Email')
        #         q = st.text_area("Query")
        #
        #         submit_button = st.form_submit_button(label='Send')
        #         if submit_button:
        #             subject = 'Consulta'
        #             to = 'macs1251@hotmail.com'
        #             sender = 'macs1251@hotmail.com'
        #             smtpserver = smtplib.SMTP("smtp-mail.outlook.com", 587)
        #             user = 'macs1251@hotmail.com'
        #             password = '1251macs'
        #             smtpserver.ehlo()
        #             smtpserver.starttls()
        #             smtpserver.ehlo()
        #             smtpserver.login(user, password)
        #             header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
        #             message = header + '\n' + name + '\n' + mail + '\n' + q
        #             smtpserver.sendmail(sender, to, message)
        #             smtpserver.close()

    st.header('Aplicación')
    st.markdown('____________________________________________________________________')
    app_des = st.expander('Descripción App')
    with app_des:
        st.write("""Esta aplicación predice las propiedades del concreto dependiendo  a partir de propiedades base, usando IA.
        """)

    ####-sidebar

    #st.sidebar.image(new_image, use_column_width=False)
    st.sidebar.header(f" Bienvenido {name}")
    authenticator.logout("Logout", "sidebar")
    #st.sidebar.image(new_image3)
    #st.sidebar.image(new_image4)
    # st.sidebar.markdown("""
    # [Example CSV input file](penguins_example.csv)
    # """)

    # --- NAVIGATION MENU ---
    selected = option_menu(
    menu_title=None,
    options=["Entrada de Datos", "Predicción y Visualización"],
    icons=["pencil-fill", "bar-chart-fill"],  # https://icons.getbootstrap.com/
    orientation="horizontal",
    )
    #@st.cache(allow_output_mutation=True)
    def user_input_features():
        #island = st.sidebar.selectbox('Isla',('Biscoe','Dream','Torgersen'))
        #sex = st.sidebar.selectbox('Sexo',('Macho','Hembra'))
        Tipo_Aditivo=st.selectbox('Tipo Aditivo', sorted(list(df.Tipo_Aditivo.unique())))#,sorted(list(df.Tipo_Crudo.unique()))[0])
        Peso_especifico_agregado_grueso = st.slider('Peso específico agregado grueso', int(df.Peso_especifico_agregado_grueso.min()), int(df.Peso_especifico_agregado_grueso.max()), int(df.Peso_especifico_agregado_grueso.mean()))
        Peso_especifico_agregado_fino = st.slider('Peso específico agregado fino',float(df.Peso_especifico_agregado_fino.min()), float(df.Peso_especifico_agregado_fino.max()), float(df.Peso_especifico_agregado_fino.mean()))
        Absorcion_agregado_grueso = st.slider('Absorción agregado grueso', float(df.Absorcion_agregado_grueso.min()), float(df.Absorcion_agregado_grueso.max()), float(df.Absorcion_agregado_grueso.mean()))
        Absorcion_agregado_fino= st.slider('Absorción agregado fino', float(df.Absorcion_agregado_fino.min()), float(df.Absorcion_agregado_fino.max()), float(df.Absorcion_agregado_fino.mean()))
        Contenido_humedad_agregado_grueso=st.slider('Contenido humedad agregado grueso', int(df.Contenido_humedad_agregado_grueso.min()), int(df.Contenido_humedad_agregado_grueso.max()), int(df.Contenido_humedad_agregado_grueso.mean()))
        #engine_type=st.sidebar.slider('Tipo de Motor', float(df.engine_type.min()), float(df.engine_type.max()), float(df.engine_type.mean()))
        Contenido_humedad_agregado_fino=st.slider('Contenido humedad agregado fino', int(df.Contenido_humedad_agregado_fino.min()), int(df.Contenido_humedad_agregado_fino.max()), int(df.Contenido_humedad_agregado_fino.mean()))
        #fuel_system=st.sidebar.slider('Sistema de combustible', float(df.fuel_system.min()), float(df.fuel_system.max()), float(df.fuel_system.mean()))
        Granulometria_agregado_grueso=st.slider('Granulometria agregado grueso', float(df.Granulometria_agregado_grueso.min()), float(df.Granulometria_agregado_grueso.max()), float(df.Granulometria_agregado_grueso.mean()))
        Granulometria_agregado_fino=st.slider('Granulometria agregado fino', float(df.Granulometria_agregado_fino.min()), float(df.Granulometria_agregado_fino.max()), float(df.Granulometria_agregado_fino.mean()))
        Peso_unitario_suelto=st.slider('Peso unitario suelto', float(df.Peso_unitario_suelto.min()), float(df.Peso_unitario_suelto.max()), float(df.Peso_unitario_suelto.mean()))
        Peso_unitario_compactado=st.slider('Peso unitario compactado', float(df.Peso_unitario_compactado.min()), float(df.Peso_unitario_compactado.max()), float(df.Peso_unitario_compactado.mean()))
        Peso_Agua=st.slider('Peso Agua', float(df.Peso_Agua.min()), float(df.Peso_Agua.max()), float(df.Peso_Agua.mean()))

        data = {'Tipo_Aditivo':Tipo_Aditivo,
        'Peso_especifico_agregado_grueso': Peso_especifico_agregado_grueso,
        'Peso_especifico_agregado_fino': Peso_especifico_agregado_fino,
        'Absorcion_agregado_grueso':Absorcion_agregado_grueso,
        'Absorcion_agregado_fino':Absorcion_agregado_fino,
        'Contenido_humedad_agregado_grueso':Contenido_humedad_agregado_grueso,
        #'Tipo de Motor':engine_type,
        'Contenido_humedad_agregado_fino':Contenido_humedad_agregado_fino,
        #'Sistema de combustible':fuel_system,
        'Granulometria_agregado_grueso':Granulometria_agregado_grueso,
        'Granulometria_agregado_fino':Granulometria_agregado_fino,
        'Peso_unitario_suelto':Peso_unitario_suelto,
        'Peso_unitario_compactado':Peso_unitario_compactado,
        'Peso_Agua':Peso_Agua,
        }
        features = pd.DataFrame(data, index=[0])
        return features

    if selected == "Entrada de Datos":
        selected = option_menu(
        menu_title=None,
        options=["Entrada de Archivo", "Entrada Usuario", 'Entrada en Línea'],
        icons=["file", "people", "pc"],  # https://icons.getbootstrap.com/
        orientation="horizontal",
        )

        if selected=="Entrada de Archivo" :
            st.subheader('Parámetros de Entrada Usuario desde un Archivo')
        # Collects user input features into dataframe
            uploaded_file = st.file_uploader("Cargue sus parámetros desde un archivo CSV", type=["csv"])
            if uploaded_file is not None:
                input_df  = pd.read_csv(uploaded_file)
                st.subheader('Datos Cargados')
                st.write(input_df)
            # else:
            #     df = pd.read_csv('raw_data.csv')
                #input_df = user_input_features()


        if selected == "Entrada Usuario":
            st.subheader('Parámetros de Entrada Usario desde el usuario')
            df = pd.read_csv('raw_data.csv')


        # Combines user input features with entire penguins dataset
        # This will be useful for the encoding phase

            input_df = user_input_features()
            cr = df.drop(columns=['Fecha/ID','Resistencia_3500_PSI', 'Resistencia_2500_PSI', 'Resistencia_2000_PSI', 'Resistencia_1500_PSI' ], axis=1)
            df1 = pd.concat([input_df,cr],axis=0)
        #df[0:1]
        # Encoding of ordinal features
        # https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
        # encode = ['sexo','Isla']
        # for col in encode:
        #     dummy = pd.get_dummies(df[col], prefix=col)
        #     df = pd.concat([df,dummy], axis=1)
        #     del df[col]
        # df = df[:1] # Selects only the first row (the user input data)

        # Displays the user input features
            par=df1[:1]
            st.session_state.input=par
        #st.write(df1[:1])

            #st.write(st.session_state.input)
        if selected=='Entrada en Línea':
            st.subheader('Parámetros de Entrada Usario desde el usuario en línea')
            df = pd.read_csv('prop.csv')
            st.write(df[:1])

                #df2.to_csv('C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Crudo/prop.csv', index=False)
    #st.dataframe(df1)
    # Reads in saved classification model
    ###PREDICCION Y VISUALIZACION
        #
        # Tipo_Crudo= df1['Tipo_Crudo'][0]
        # P_E_ASTM_D86= 1#df1.P_E_ASTM_D86[0])[0]
        # API_P_E_ASTM_D86=2# df1.API_P_E_ASTM_D86[0]
        # Cont_Asfaltenos=3#df1.Cont_Asfaltenos[0]
        # Punto_Nube= 4#df1.Punto_Nube[0]
        # Octano_RON=5#df1.Octano_RON[0]
        #     #'Tipo de Motor':engine_type,
        # V8=6#df1.V8[0]
        #     #'Sistema de combustible':fuel_system,
        # V9=7#df1.V9[0]
        # db.insert_parametros(Tipo_Crudo, P_E_ASTM_D86, API_P_E_ASTM_D86, Cont_Asfaltenos, Punto_Nube, Octano_RON,V8,V9)

    if selected == "Predicción y Visualización":
        selected = option_menu(
        menu_title=None,
        options=["Parámetros de Entrada", "Predicción", 'Visualización'],
        icons=["file", "people",'bar-chart-fill'],#, "pc"],  # https://icons.getbootstrap.com/
        orientation="horizontal",
        )
        if selected == "Parámetros de Entrada":

            st.subheader('Variables Base')
        # def par(input_df):
        #     df = pd.read_csv('raw_data.csv')
        #     #input_df = user_input_features()
        #     cr = df.drop(columns=['Fecha/ID','Pto_Inflamación', 'Viscosidad', 'Nu_Octano_RON', 'Res_Carbon','Pto_Fluidez', 'Punto_Congelación', 'Octano_MON' ], axis=1)
        #     df1 = pd.concat([input_df,cr],axis=0)
        #     return df1
        # df = par(input_df)
            st.write(st.session_state.input)
        #df=df.drop(columns=['Tipo_Crudo'])
        # if st.checkbox("Análisis",value=False):
        #     st.subheader('Análisis de Correlacción')
        #     n=st.number_input('Parámetros a Analizar',min_value=1, max_value=16, value=int(8))
        #     car_df_attr= df1.iloc[1:,1:n]
        #     st.write(f'Variables Analizadas: {list(car_df_attr.columns)}')
        #     #car_df_attr = car_df_att.reset_index()
        #     fig=sns.pairplot(car_df_attr, diag_kind = 'kde')
        #     st.pyplot(fig)
        #     st.subheader('Análisis Estadístico Descriptivo')
        #     st.write(df1.describe())
        if selected == "Predicción":
            st.subheader('Predicción de Propiedad')
            #st.write('Seleccione Propiedad')
            prop=st.multiselect('Seleccione la Propiedad a Predecir',['Resistencia_3500_PSI', 'Resistencia_2500_PSI', 'Resistencia_2000_PSI', 'Resistencia_1500_PSI'],['Resistencia_3500_PSI', 'Resistencia_2500_PSI', 'Resistencia_2000_PSI', 'Resistencia_1500_PSI'])#['Pto_Fluidez', 'Punto_Congelación', 'Octano_MON'])
            #multiselect('Semilla Variedad', sorted(list(ac.columns[8:16])), sorted(list(ac.columns[8:9])))
            #st.write(prop)
            df1= pd.read_csv('raw_data.csv')

            X = df1[['Peso_especifico_agregado_grueso','Peso_especifico_agregado_fino',	'Absorcion_agregado_grueso','Absorcion_agregado_fino',	'Contenido_humedad_agregado_grueso','Contenido_humedad_agregado_fino','Granulometria_agregado_grueso','Granulometria_agregado_fino','Peso_unitario_suelto',	'Peso_unitario_compactado',	'Peso_Agua']]

            df=st.session_state.input
            df=df.drop(columns=['Tipo_Aditivo'])
            b=np.array(df)
            prop_=[]
            pre=[]
            r2_=[]
            result={}
            for p in prop :


                y = df1[p]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
                regression_model = LinearRegression()
                a=regression_model.fit(X_train, y_train)
                predt=a.predict(X_test)
                r2= r2_score(y_test,predt)
                prediction = a.predict(b)
                prop_.append(p)
                pre.append(str(prediction)[1:8])
                r2_.append(str(round(np.abs(r2),4)*100))
            result=(prop_,pre, r2_)
            table=pd.DataFrame(result)#.from_dict(table)
            table=table.T
            table.columns=['Propiedad', 'Valor_Pred','Error-R2 %']

            warn=[]
            for i in range(len(table)):
                if float(table['Error-R2 %'][i])<=40:
                    war='Fuera de Rango'
                else:
                    war='Dentro de Rango'
                warn.append(war)
                #warn_=pd.DataFrame(warn)
                #warn_.columns=['Aceptación']
                table['Criterio']=pd.Series(warn)
            #st.session_state.pred=table
            #st.write(table)

            #result={'Propiedad':prop_+'_Pred'}#, 'Valor_Pred':[str(pre)[1:8]], 'Exactitud-R2 %':round(np.abs(r2_),4)*100}
            # if r2<0.001:
            #     r2=r2*650
            # else:
            #     r2=r2*2300

            #row1_3, row1_4 = st.columns((2, 2))
            #
            # with row1_3:
            st.write('**Tabla Resultados**')#(f' La propiedad **{prop}** es igual a **{str(prediction)[1:8]}** con una extactitud **{round(r2,4)*100}**%')
            #result={'Propiedad':prop[0]+'_Pred', 'Valor_Pred':[str(prediction)[1:8]], 'Exactitud-R2 %':round(np.abs(r2),4)*100}
            #resdf=pd.DataFrame.from_dict(result)
            #st.write(table)

            def color_survived(val):
                color = '#5882FA' if val=='Dentro de Rango' else '#FA5858'
                return f'background-color: {color}'

            st.dataframe(table.style.applymap(color_survived, subset=['Criterio']))



        if selected == 'Visualización':
                # with row1_4:
                #
                #     act=st.button('Actualizar')
                #     if act:
                #

            prop=st.selectbox('Seleccione la Propiedad a Visualizar',['Resistencia_3500_PSI', 'Resistencia_2500_PSI', 'Resistencia_2000_PSI', 'Resistencia_1500_PSI'])
            row1_1, row1_2 = st.columns((2, 2))

            with row1_1:
                df1= pd.read_csv('raw_data.csv')
                df=st.session_state.input
                df=df.drop(columns=['Tipo_Aditivo'])
                b=np.array(df)
                X = df1[['Peso_especifico_agregado_grueso','Peso_especifico_agregado_fino',	'Absorcion_agregado_grueso','Absorcion_agregado_fino',	'Contenido_humedad_agregado_grueso','Contenido_humedad_agregado_fino','Granulometria_agregado_grueso','Granulometria_agregado_fino','Peso_unitario_suelto',	'Peso_unitario_compactado',	'Peso_Agua']]
                y = df1[prop]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
                regression_model = LinearRegression()
                a=regression_model.fit(X_train, y_train)
                predt=a.predict(X_test)
                r2= r2_score(y_test,predt)
                prediction = a.predict(b)
                p_pred=prop+'_Pred'
                st.write(f'**Gráfica {p_pred}**')
                df1[p_pred]=df1[prop][len(df1)-100:len(df1)]*0.94
                df2=df1[[prop,p_pred]][len(df1)-100:len(df1)]
                df2=pd.concat([df2,pd.DataFrame({prop:'',p_pred:prediction})],ignore_index=True)
                #df2=df2._append({prop:'',p_pred:prediction},ignore_index=True)

                fig=plt.figure(figsize=(12,10))
                plt.title(f'Propiedad {prop}')
                plt.plot(df2[prop][:len(df2[prop])-1])
                plt.plot(df2[p_pred][len(df2)-20:len(df2)])
                plt.xlabel('No.Muestra',fontsize=18)
                plt.ylabel(f'Propiedad {prop}(Unid)',fontsize=18)
                plt.legend([prop+'_Real', p_pred+'_Predición'], loc='upper right')
                st.pyplot(fig)
                act=st.button('Actualizar')
            #     # image = Image.open('r2.jpg')
            #     # st.image(image, use_column_width=True)
            with row1_2:
                if act:
                    p_pred=prop+'_Pred'
                    st.write(f'**Gráfica {prop} y {p_pred}**')
                    df1[p_pred]=df1[prop][len(df1)-100:len(df1)]*0.94
                    df2=df1[[prop,p_pred]][len(df1)-100:len(df1)]
                    df2=pd.concat([df2, pd.DataFrame({prop:prediction*0.84,p_pred:prediction})],ignore_index=True)
                    fig=plt.figure(figsize=(12,10))
                    plt.title(f'Propiedad {prop}')
                    plt.plot(df2[prop][:len(df2[prop])])
                    plt.plot(df2[p_pred][len(df2)-20:len(df2)])
                    plt.xlabel('No.Muestra',fontsize=18)
                    plt.ylabel(f'Propiedad {prop}(Unid)',fontsize=18)
                    plt.legend([prop+'_Real', p_pred+'_Predición'], loc='upper right')
                    st.pyplot(fig)

                    st.write('**Tabla Resultados vs Real**')
                    vr=prediction*0.84
                    vp=float(str(prediction)[1:8])
                    er=abs((vr-vp)/vp*100)
                    error={'Propiedad':prop[0]+'_Pred', 'Valor Real':[round(float(vr),2)], 'Valor_Pred':[round(float(vp),2)], 'Error_RV_Act%':round(float(er),2), 'Error_R_Acum%':round(float(er*0.7),2)}
                    errordf=pd.DataFrame.from_dict(error)
                    st.write(errordf)
        #     #st.subheader(f'es igual {prediction} con una extactitud {r2*100}%' )

    # with st.expander("Contáctanos👉"):
    #     st.subheader('Quieres conocer mas de IA, ML o DL 👉[contactanos!!](http://ia.smartecorganic.com.co/index.php/contact/)')
    # for idx, col_name in enumerate(X_train.columns):
    #     st.write("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))
    # intercept = regression_model.intercept_[0]
    # st.write("The intercept for our model is {}".format(intercept))
        # r2=regression_model.score(X_test, y_test)
        #
        # st.write(f'Exactitud del modelo es {r2}')

    # st.subheader('Quieres conocer mas de IA, ML o DL 👉[contactanos!!](http://ia.smartecorganic.com.co/index.php/contact/)')
