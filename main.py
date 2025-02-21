import mli.flickr_transformer.main as st
import torch
from PIL import Image
import io
import utils
import infer

st.title('Image Caption Generator')

# Load model (cache this)
@st.cache_resource
def load_models():
    return infer.models()

model, clip_model, clip_processor = load_models()

# Image input section
st.header("Upload an Image")
img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process whichever image is provided
final_image = img_file

if final_image:
    # Display image
    image = Image.open(io.BytesIO(final_image.getvalue()))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Generate caption
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            
            predictions = []

            # Generate captions
            for k in range(5):
                caption = infer.generate_caption(image)
                print(f"Caption {k}: {caption}")
                predictions.append(caption)

            processor_output = clip_processor(
                images=image, 
                text=predictions, 
                return_tensors="pt", 
                padding=True, 
                do_rescale=True
            ).to(utils.get_device())
            model_output = clip_model(**processor_output)
            best_idx = torch.argmax(model_output.logits_per_image).item()
            best_prediction = predictions[best_idx]
            best_caption = infer.format_caption(best_prediction)
            st.success(f"{best_caption}")


