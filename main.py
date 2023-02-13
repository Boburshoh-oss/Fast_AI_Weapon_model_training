import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
if plt == 'linux':pathlib.WindowsPath = pathlib.PosixPath\

st.set_page_config(
    page_title="Weapon vs Mobile Phone",
    page_icon="🔫",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# Sidebar
with st.sidebar:
    st.write('---')

    st.title("Resurslar")
    st.header('Modellar')
    st.write('''
        - <a href="https://drive.google.com/file/d/12RFWWE7lI3Sj59xhDmuykiT2VFwudwsE/view?usp=sharing" style="color:green;" target="_blank">filter model.pkl</a>
        - <a href="https://drive.google.com/file/d/1vLZGWXzvKUwc2fYHbSJwj37bNde9_0S2/view?usp=sharing" style="color:green;" target="_blank">weapon vs mobile phone.pkl</a>
    ''', unsafe_allow_html=True)

    st.header('Train qilingan colablar:')
    st.write('''
        - <a href="https://github.com/Boburshoh-oss/Fast_AI_Weapon_model_training" style="color:green;" target="_blank">Github Repository</a>
    ''', unsafe_allow_html=True)

    

#title
st.title('"Weapon vs Mobile Phone" klassifikatsiya qiluvchi model')

#model 
model = load_learner('weapon_model.pkl')
filter_model = load_learner("filter_model.pkl")



with st.expander("Qo'llanma"):
    st.write('''
        Loyihada ikkita DL modeli train va deploy qilindi.
        Ushbu dastur bir necha xil qurollar va Mobile Telfonlarni tanib beruvchi dastur:
        - Missile (Raketa)
        - Handgun (To'pponcha)
        - Rifle (Miltiq)
        - Tank 
        - Mobile Phone (Mobil telfonlar)
    ''')
    st.info('*Resurslarni chap yuqoridagi tugmani bosish orqali saydbardan topishingiz mumkin*.')
    st.warning("Agar siz yuqoridagi obyektlarga aloqador rasmlar yuborsangiz dastur adashish ehtimoli bor. Misol uchun Mashina, dastur buni qurol deb aniqlashi mumkin sabab: o'qitilgan qurol rasmlar orasida mashina ham bo'lishi mumkin!", icon='⚠️')

st.write('---')

#rasmni joylash
upload = st.file_uploader("Rasmingizni yuklang", type=['png','jpeg','jpg','gif','svg'])

camera_photo = st.camera_input("Rasmga oling")

def indintify(img):
    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik:{probs[pred_id]*100:.1f}%")

    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)

def check_related(upload):
    st.image(upload)
    #PIL convert
    img = PILImage.create(upload)

    #prediction
    pred, pred_id, probs = filter_model.predict(img)
    if pred=="unrelated":
        st.warning("Rasm yuqoridagi obyektlarga mos emas! Iltimos Qo'llanmani o'qib chiqing")
    else:
        indintify(img)
        

if upload is not None:
    check_related(upload)
if camera_photo is not None:
    check_related(camera_photo)