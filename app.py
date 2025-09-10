import streamlit as st
import tensorflow as tf
import numpy as np
import os, json
from PIL import Image


MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "egypt_cnn.h5")
CLASS_MAP_PATH = os.path.join(MODELS_DIR, "class_indices.json")
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    return model, id2label


info_dict = {
    "Akhenaten": "إخناتون (1353–1336 ق.م) الملك الذي دعا لعبادة آتون وبنى عاصمته أخيتاتون (تل العمارنة).",
    "AmenhotepIII": "أمنحتب الثالث (1391–1353 ق.م) ازدهرت مصر في عصره وبنى معبد الأقصر والعديد من التماثيل الضخمة.",
    "Bent pyramid for senefru": "الهرم المنحني في دهشور بناه الملك سنفرو (2600 ق.م تقريبًا) وهو أول محاولة لبناء هرم كامل.",
    "Colossoi of Memnon": "تمثالا ممنون هما تمثالان ضخمان للفرعون أمنحتب الثالث في الأقصر.",
    "Goddess Isis": "الإلهة إيزيس رمز الأمومة والسحر والحماية عند قدماء المصريين.",
    "Hatshepsut face": "الملكة حتشبسوت (1479–1458 ق.م) أشهر ملكة فرعونية وبنت معبدها الجنائزي بالدير البحري.",
    "Khafre Pyramid": "هرم خفرع ثاني أكبر أهرامات الجيزة بُني حوالي 2570 ق.م.",
    "King Thutmose III": "تحتمس الثالث (1479–1425 ق.م) أعظم القادة العسكريين ووسع حدود مصر لأبعد مدى.",
    "Mask of Tutankhamun": "قناع توت عنخ آمون الذهبي اكتشف عام 1922 ويُعد من أهم الآثار المصرية.",
    "Nefertiti": "نفرتيتي زوجة إخناتون وأيقونة الجمال المصري القديم.",
    "Pyramid_of_Djoser": "هرم زوسر المدرج (2670 ق.م) أول هرم في التاريخ بناه إمحوتب.",
    "Ramesses II": "رمسيس الثاني (1279–1213 ق.م) أعظم فراعنة مصر، صاحب معركة قادش وباني معبد أبو سمبل.",
    "Ramessum": "الرامسيوم هو المعبد الجنائزي لرمسيس الثاني بالأقصر.",
    "Statue of King Zoser": "تمثال الملك زوسر مؤسس الأسرة الثالثة وباني الهرم المدرج.",
    "Statue of Tutankhamun with Ankhesenamun": "تمثال لتوت عنخ آمون وزوجته عنخس إن آمون يرمز إلى الحب الملكي.",
    "Temple_of_Hatshepsut": "معبد حتشبسوت بالدير البحري تحفة معمارية من الأسرة 18.",
    "Temple_of_Isis_in_Philae": "معبد إيزيس في فيلة بأسوان كان مركز عبادة الإلهة إيزيس حتى العصر الروماني.",
    "Temple_of_Kom_Ombo": "معبد كوم أمبو مكرس للإلهين سوبك وحورس.",
    "The Great Temple of Ramesses II": "معبد أبو سمبل العظيم بناه رمسيس الثاني تخليداً لانتصاراته.",
    "menkaure pyramid": "هرم منكاورع أصغر أهرامات الجيزة بُني حوالي 2510 ق.م.",
    "sphinx": "أبو الهول تمثال ضخم من الحجر الجيري يُجسد وجه الملك خفرع وجسد أسد."
}


st.title("Ancient Egypt Monuments Classifier")
st.caption("trained to tell you which ancient Egyptian monument you asked for!")


if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASS_MAP_PATH)):
    st.error("❌ الموديل أو ملف الكلاسات مش موجودين في مجلد models/")
    st.stop()

model, id2label = load_model_and_labels()


uploaded_file = st.file_uploader("upload your image ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="your image", use_column_width=True)

    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_id = np.argmax(preds)
    confidence = np.max(preds)
    label = id2label[class_id]

    st.success(f"result : **{label}** ")

    st.subheader("ℹ️ معلومات تاريخية:")
    st.write(info_dict.get(label, "لا توجد معلومات متاحة."))
