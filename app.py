"""Start with streamlit run app.py."""
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sonicclassifier.audio import GENRES
from sonicclassifier.inference import load_checkpoint, predict
from sonicclassifier.model import get_device

st.set_page_config(page_title='SonicClassifier', page_icon='🎵')
st.title('🎵 SonicClassifier')
st.write('Explore how a CNN reads music through mel spectrograms.')
st.caption('No pretrained model is distributed. Train a checkpoint before predicting. Probabilities are model scores, not calibrated confidence.')
checkpoint = st.sidebar.text_input('Local checkpoint path', 'artifacts/run/best.pt')
upload = st.file_uploader('Upload music (up to 20 MB)', type=['wav', 'mp3', 'flac', 'ogg', 'au'])
if upload is not None:
    if upload.size > 20 * 1024 * 1024:
        st.error('Please upload a clip smaller than 20 MB.')
    elif not Path(checkpoint).is_file():
        st.info('Train a model first, then enter its best.pt path in the sidebar.')
    else:
        st.audio(upload)
        try:
            device = get_device()
            model, config, _ = load_checkpoint(checkpoint, device)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ('upload' + Path(upload.name).suffix)
                path.write_bytes(upload.getvalue())
                with st.spinner('Computing spectrograms and predictions…'):
                    probabilities, feature = predict(path, model, config, device)
            st.subheader(f'Predicted genre: {GENRES[int(probabilities.argmax())]}')
            st.caption(f'Device: {device}. Scores average all non-overlapping segments; the last is padded.')
            st.bar_chart(pd.DataFrame({'Probability': probabilities}, index=GENRES))
            fig, ax = plt.subplots(figsize=(9, 4))
            image = ax.imshow(feature, origin='lower', aspect='auto', extent=[0, config.seconds, 0, config.n_mels], cmap='magma', vmin=0, vmax=1)
            ax.set(xlabel='Time (seconds)', ylabel='Mel band', title='First segment: normalized log-mel spectrogram')
            fig.colorbar(image, ax=ax, label='Normalized intensity')
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f'Could not analyze this file: {exc}')

with st.expander('Learn: how the CNN works', expanded=True):
    st.markdown('''
**From audio to an image.** A short-time Fourier transform measures frequencies in overlapping
windows. Mel filters combine frequencies on a scale inspired by hearing. Log power compresses
the dynamic range. Time runs horizontally, mel bands vertically, and color represents intensity.

**Convolution and filters.** Each learned 3×3 filter slides across this image. Its weights respond
to local patterns such as harmonic stacks, transients, or sustained bands. Multiple filters learn
different patterns; deeper layers combine them. ReLU keeps positive activations and adds nonlinearity.

**Pooling.** Max pooling keeps the strongest activation in each 2×2 region, reducing resolution
and making small shifts less disruptive. Global average pooling summarizes each final feature map.
A linear layer produces ten logits; softmax converts them to probabilities summing to one.

**Backpropagation.** Cross entropy compares logits with the known genre. The chain rule computes
how each weight contributed to the loss. Adam updates weights using these gradients. Repeating
this over shuffled training batches helps filters learn useful patterns instead of hand-coded rules.
Dropout regularizes the classifier. Validation selects the best epoch; held-out test songs measure
generalization only after selection.

**Honest evaluation.** All segments of a song stay in one split. GTZAN contains known duplicates,
artist overlap, and labeling issues, so song-level splitting alone does not guarantee artist-level
independence. Genre is subjective, and unfamiliar or mixed-genre audio can produce misleading scores.
''')
