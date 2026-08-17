import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import io

# ================================
# Initialize Flask App
# ================================
app = Flask(__name__)

# ================================
# Configuration
# ================================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================================
# Load VGG16 Transfer Learning Model
# ================================
print("Loading VGG16 Transfer Learning Model...")
model = keras.models.load_model('best_transfer_model.keras')
print("Model loaded! ✅")
print("Accuracy: 90.06% — VGG16 Transfer Learning!")

# Store last result for PDF
last_result = {}

# ================================
# Helper Functions
# ================================
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_xray(img_path):
    try:
        # Load image — RGB for VGG16!
        img = image.load_img(
            img_path,
            target_size=(150, 150),
            color_mode='rgb'        # ✅ RGB for VGG16!
        )
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)
        probability = float(prediction[0][0])

        if probability > 0.5:
            result = "PNEUMONIA"
            confidence = round(probability * 100, 2)
            risk = "HIGH RISK"
            recommendation = "Immediate medical attention required! Please consult a doctor immediately."
        else:
            result = "NORMAL"
            confidence = round((1 - probability) * 100, 2)
            risk = "LOW RISK"
            recommendation = "Lungs appear normal. Regular checkup advised."

        return {
            "result": result,
            "confidence": confidence,
            "risk": risk,
            "recommendation": recommendation
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}

def generate_pdf(patient_data, result_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title Style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=10,
        alignment=1
    )

    # Subtitle Style
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.grey,
        alignment=1,
        spaceAfter=20
    )

    # Header
    story.append(Paragraph("🏥 MEDICAL AI REPORT", title_style))
    story.append(Paragraph(
        "Pneumonia Detection — Chest X-Ray Analysis",
        subtitle_style))
    story.append(Paragraph(
        "Powered by VGG16 Transfer Learning — 90.06% Accuracy",
        subtitle_style))
    story.append(Spacer(1, 0.2*inch))

    # Date
    date_str = datetime.now().strftime("%d %B %Y | %I:%M %p")
    story.append(Paragraph(
        f"Report Generated: {date_str}",
        styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Patient Details Table
    story.append(Paragraph("PATIENT DETAILS", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))

    patient_table_data = [
        ['Field', 'Details'],
        ['Patient Name', patient_data.get('name', 'N/A')],
        ['Age', patient_data.get('age', 'N/A')],
        ['Gender', patient_data.get('gender', 'N/A')],
        ['Doctor Name', patient_data.get('doctor', 'N/A')],
        ['Scan Date', date_str],
    ]

    patient_table = Table(patient_table_data,
                         colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0),
         colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (0,-1),
         colors.HexColor('#f0f0ff')),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.HexColor('#f9f9f9')]),
    ]))

    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))

    # AI Result
    story.append(Paragraph("AI DIAGNOSIS RESULT", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))

    result = result_data.get('result', 'N/A')
    confidence = result_data.get('confidence', 0)
    risk = result_data.get('risk', 'N/A')
    recommendation = result_data.get('recommendation', 'N/A')

    if result == 'PNEUMONIA':
        result_color = colors.red
    else:
        result_color = colors.green

    result_style = ParagraphStyle(
        'Result',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=result_color,
        alignment=1,
        spaceAfter=10
    )

    story.append(Paragraph(result, result_style))

    result_table_data = [
        ['Metric', 'Value'],
        ['Diagnosis', result],
        ['Confidence Score', f"{confidence}%"],
        ['Risk Level', risk],
        ['AI Model', 'VGG16 Transfer Learning'],
        ['Model Accuracy', '90.06%'],
        ['Recommendation', recommendation],
    ]

    result_table = Table(result_table_data,
                        colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0),
         colors.HexColor('#764ba2')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (0,-1),
         colors.HexColor('#f0f0ff')),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.HexColor('#f9f9f9')]),
    ]))

    story.append(result_table)
    story.append(Spacer(1, 0.3*inch))

    # Disclaimer
    story.append(Paragraph("⚠️ DISCLAIMER", styles['Heading2']))
    story.append(Paragraph(
        "This report is generated by an AI system for "
        "educational purposes only. Always consult a "
        "qualified medical professional for diagnosis "
        "and treatment decisions.",
        styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph(
        "Powered by VGG16 Transfer Learning Medical AI — "
        "Built by  Chinthala Karunakar | ECE Student ",
        footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ================================
# Routes
# ================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global last_result
    try:
        # Get patient details
        patient_data = {
            'name': request.form.get('patient_name', 'Unknown'),
            'age': request.form.get('patient_age', 'Unknown'),
            'gender': request.form.get('patient_gender', 'Unknown'),
            'doctor': request.form.get('doctor_name', 'Unknown'),
        }

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded!"})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected!"})

        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result = predict_xray(filepath)
            result['patient'] = patient_data

            # Store for PDF
            last_result = {
                'patient': patient_data,
                'result': result
            }

            print(f"Patient: {patient_data}")
            print(f"Result: {result}")

            return jsonify(result)

        return jsonify({"error": "Invalid file type!"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})

@app.route('/download_pdf')
def download_pdf():
    global last_result
    try:
        if not last_result:
            return "No result available!", 400

        pdf_buffer = generate_pdf(
            last_result['patient'],
            last_result['result']
        )

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"medical_report_{last_result['patient']['name']}.pdf"
        )

    except Exception as e:
        print(f"PDF Error: {e}")
        return f"Error generating PDF: {e}", 500

# ================================
# Run App
# ================================
if __name__ == '__main__':
    print("\n" + "=" * 45)
    print("   MEDICAL AI WEB APP — VGG16 UPGRADED!")
    print("=" * 45)
    print("Model: VGG16 Transfer Learning")
    print("Accuracy: 90.06%")
    print("Open → http://localhost:5000")
    print("=" * 45 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)