from flask import Flask, render_template, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load Hugging Face models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
sentiment_analyzer = pipeline("sentiment-analysis")

# Route to render the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and generate context and sentiment
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Load image
        image = Image.open(file.stream)

        # Generate caption (context) from the image
        inputs = caption_processor(images=image, return_tensors="pt")
        out = caption_model.generate(**inputs)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)

        # Perform sentiment analysis on the caption
        sentiment = sentiment_analyzer(caption)

        return jsonify({
            'caption': caption,
            'sentiment': sentiment[0]['label'],
            'confidence': sentiment[0]['score']
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
