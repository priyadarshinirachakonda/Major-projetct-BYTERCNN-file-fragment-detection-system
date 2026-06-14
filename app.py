# import streamlit as st
# import subprocess
# import tempfile
# import os
# import time
# import sys
# time.sleep(1)


# st.set_page_config(page_title="File Fragment Classifier", layout="centered")

# st.title("📂 File Fragment Type Detection")
# st.write("Upload a file fragment to predict its file type")
# st.write("Model works on raw byte content (file extension is ignored).")

# # Inputs
# model_path = "output_sc2/bytercnn_len4096_sc2.keras"
# scenario = "2"

# uploaded_file = st.file_uploader("Upload a file", type=None)

# if uploaded_file is not None:
#     # Save uploaded file temporarily
#     with tempfile.NamedTemporaryFile(delete=False) as tmp:
#         tmp.write(uploaded_file.read())
#         temp_file_path = tmp.name

#     st.success("File uploaded successfully!")

#     if st.button("Predict File Type"):
#         with st.spinner("Running deep learning model (this may take some time)..."):
            
#             result = subprocess.run(
#                 [sys.executable, "predict.py", model_path, scenario, temp_file_path],
#                 capture_output=True,
#                 text=True
#             )
#             capture_output=True,
#             text=True,
#             env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}


#         st.subheader("🔍 Prediction Result")
#         if result.stdout:
#             st.text(result.stdout)
#         else:
#             st.error("No output received!")
#             st.text(result.stderr)


#     os.remove(temp_file_path)


import streamlit as st
import tempfile
import subprocess
import os
from predict_hierarchical import predict_hierarchical

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="ByteRCNN File Type Predictor",
    layout="centered"
)

# =========================================================
# TITLE
# =========================================================

st.title("📂 ByteRCNN File Type Predictor")

st.write(
    "Upload any file to predict its type "
    "using deep learning on random byte fragments."
)

# =========================================================
# FILE UPLOAD
# =========================================================

st.markdown(
    """
    <style>

    /* Hide uploaded file container */
    div[data-testid="stFileUploaderFile"] {
        display: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(

    "Upload a file",
    type=None
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # st.success(
    #     f"Uploaded: {uploaded_file}"
    # )
    st.success("File uploaded successfully")

    # ---------------------------------------------
    # SAVE TEMP FILE
    # ---------------------------------------------

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=os.path.splitext(
            uploaded_file.name
        )[1]

    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        temp_path = tmp.name

    # ---------------------------------------------
    # PREDICT BUTTON
    # ---------------------------------------------

    if st.button("Predict File Type"):

        with st.spinner(
            "Analyzing byte fragments..."
        ):

            try:

                prediction = predict_hierarchical(
                    temp_path
                )

                st.subheader(
                    "Prediction Result"
                )

                st.code(prediction)

            except Exception as e:

                st.error(str(e))

    # ---------------------------------------------
    # DELETE TEMP FILE
    # ---------------------------------------------

    try:

        os.remove(temp_path)

    except:

        pass

st.info(

    """
    ⚠️ Note:

    This system performs file type identification using
    deep learning on raw byte fragments.

    The prediction process does NOT rely on:

    • File extensions  
    • File headers / magic bytes  
    • Metadata signatures  
    • File names  

    Instead, the model analyzes internal binary byte
    patterns extracted from random file fragments.
    """
)