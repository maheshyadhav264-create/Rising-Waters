const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle,
  PageBreak, ExternalHyperlink,
} = require("docx");

const ROOT = "/home/claude/floods_prediction";
const img = (p) => fs.readFileSync(path.join(ROOT, p));

const ACCENT = "1F6F78";   // deep teal
const DARK = "13293D";
const MUTED = "5B7280";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 140 },
    children: [new TextRun({ text, ...opts })],
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}
function code(text) {
  return new Paragraph({
    spacing: { after: 100 },
    shading: { type: ShadingType.CLEAR, fill: "F1F5F7" },
    children: [new TextRun({ text, font: "Consolas", size: 19 })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 260 },
    children: [new TextRun({ text, italics: true, size: 18, color: MUTED })],
  });
}
function centeredImage(buffer, width, height) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [new ImageRun({ data: buffer, transformation: { width, height }, type: "png" })],
  });
}

function twoColTable(rows) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: [3200, 6300],
    rows: rows.map(([a, b], i) =>
      new TableRow({
        children: [
          new TableCell({
            width: { size: 3200, type: WidthType.DXA },
            shading: i === 0 ? { type: ShadingType.CLEAR, fill: DARK } : undefined,
            children: [new Paragraph({ children: [new TextRun({ text: a, bold: true, color: i === 0 ? "FFFFFF" : "000000" })] })],
          }),
          new TableCell({
            width: { size: 6300, type: WidthType.DXA },
            shading: i === 0 ? { type: ShadingType.CLEAR, fill: DARK } : undefined,
            children: [new Paragraph({ children: [new TextRun({ text: b, bold: i === 0, color: i === 0 ? "FFFFFF" : "000000" })] })],
          }),
        ],
      })
    ),
  });
}

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22 } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", run: { size: 32, bold: true, color: ACCENT }, paragraph: { spacing: { before: 300, after: 160 } } },
      { id: "Heading2", name: "Heading 2", run: { size: 26, bold: true, color: DARK }, paragraph: { spacing: { before: 240, after: 120 } } },
    ],
  },
  sections: [
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        // ---------------- Title page ----------------
        new Paragraph({ spacing: { before: 1600 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "FLOODS PREDICTION", bold: true, size: 56, color: ACCENT })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
          children: [new TextRun({ text: "A Machine-Learning Web Application for Real-Time Flood Risk Forecasting", size: 26, color: DARK })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 300 },
          children: [new TextRun({ text: "Project Documentation & Results Report", size: 22, italics: true, color: MUTED })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 800 },
          children: [new TextRun({ text: "Stack: Flask · scikit-learn · joblib · HTML/CSS/JS", size: 20, color: MUTED })] }),
        new Paragraph({ children: [new PageBreak()] }),

        // ---------------- Table of contents (manual) ----------------
        h1("Contents"),
        bullet("1. Project Overview"),
        bullet("2. System Architecture"),
        bullet("3. Repository Structure"),
        bullet("4. Backend — app.py"),
        bullet("5. Frontend — templates & static assets"),
        bullet("6. Model Training Pipeline"),
        bullet("7. Setup & Run Instructions"),
        bullet("8. Results"),
        bullet("9. Conclusion & Next Steps"),
        new Paragraph({ children: [new PageBreak()] }),

        // ---------------- 1. Overview ----------------
        h1("1. Project Overview"),
        body("Flood forecasting is the use of forecasted precipitation and streamflow data, together with rainfall-runoff and streamflow-routing models, to forecast flow rates and water levels for periods ranging from a few hours to several days ahead, depending on the size of the watershed or river basin."),
        body("Flood forecasting is a key component of flood warning: forecasting builds a profile of expected channel flows or river levels at a given location, while warning is the task of using those forecasts to communicate decisions to the public. Real-time flood forecasting at a regional scale can be delivered within seconds using machine-learning models — this project implements exactly that as a small, self-contained web application."),
        body("The application accepts five seasonal weather readings from a user, scales them with a fitted StandardScaler, and passes them to a trained classification model that returns a flood / no-flood call together with a confidence score."),
        h2("Input Features"),
        twoColTable([
          ["Feature", "Description"],
          ["Cloud Cover (%)", "Percentage sky coverage — a leading indicator of incoming precipitation."],
          ["Annual Rainfall (mm)", "Total yearly precipitation — the backbone signal for basin saturation."],
          ["Jan–Feb Rainfall (mm)", "Winter-season rainfall total."],
          ["March–May Rainfall (mm)", "Pre-monsoon / spring rainfall total."],
          ["June–September Rainfall (mm)", "Monsoon-season rainfall total — typically the strongest predictor."],
        ]),

        // ---------------- 2. Architecture ----------------
        h1("2. System Architecture"),
        body("The application follows a simple three-layer design:"),
        bullet("Presentation layer — Jinja2 HTML templates styled with a single stylesheet, served by Flask."),
        bullet("Application layer — app.py, a Flask server that routes requests and orchestrates prediction."),
        bullet("Model layer — a scikit-learn classifier (floods.save) and StandardScaler (transform.save), both persisted with joblib and loaded once at startup."),
        body("Request flow: the user submits the form on /Predict → the browser POSTs the five values to /predict → app.py builds a single-row DataFrame → the row is scaled with the saved StandardScaler → the scaled row is passed to the saved model → the predicted class routes the response to chance.html (flood) or no_chance.html (safe)."),

        // ---------------- 3. Repo structure ----------------
        h1("3. Repository Structure"),
        code("floods_prediction/"),
        code("├── app.py                     # Flask application (routes + prediction logic)"),
        code("├── floods.save                # Trained classifier (joblib)"),
        code("├── transform.save             # Fitted StandardScaler (joblib)"),
        code("├── requirements.txt           # Python dependencies"),
        code("├── templates/"),
        code("│   ├── home.html              # Landing page"),
        code("│   ├── index.html             # Prediction input form"),
        code("│   ├── chance.html            # Result page — flood predicted"),
        code("│   └── no_chance.html         # Result page — no flood predicted"),
        code("├── static/"),
        code("│   ├── main.css               # Styling"),
        code("│   └── main.js                # Client-side validation"),
        code("├── model_training/"),
        code("│   ├── train_model.py         # Data generation + training + evaluation"),
        code("│   ├── synthetic_flood_data.csv"),
        code("│   └── metrics_report.txt"),
        code("└── screenshots/               # Rendered UI + results images"),

        // ---------------- 4. Backend ----------------
        h1("4. Backend — app.py"),
        body("The backend loads the trained model and scaler once at startup, then exposes four routes:"),
        twoColTable([
          ["Route", "Purpose"],
          ["GET  /", "Renders home.html — the landing / introduction page."],
          ["GET  /Predict", "Renders index.html — the input form."],
          ["POST /predict", "Reads the five form fields, builds a DataFrame, scales it, runs model.predict() / predict_proba(), and renders chance.html or no_chance.html with the confidence score."],
        ]),
        body("Core prediction snippet:"),
        code('input_df = pd.DataFrame([[cloud_cover, annual_rainfall, jan_feb_rainfall,'),
        code('                           mar_may_rainfall, jun_sep_rainfall]], columns=FEATURE_COLUMNS)'),
        code('input_scaled = sc.transform(input_df)'),
        code('prediction = model.predict(input_scaled)[0]'),
        code('probability = model.predict_proba(input_scaled)[0][1]'),

        // ---------------- 5. Frontend ----------------
        h1("5. Frontend — templates & static assets"),
        body("All four pages share a dark, water-themed visual identity (navy background, teal accent, droplet mark) defined once in static/main.css, and a consistent navbar/footer. index.html additionally uses static/main.js for lightweight client-side validation (rejecting empty, negative, or out-of-range values) before the form submits."),
        h2("Home page"),
        centeredImage(img("screenshots/home_crop.png"), 460, 259),
        caption("Landing page — introduction and call to action."),
        h2("Prediction form"),
        centeredImage(img("screenshots/index.png"), 400, 311),
        caption("index.html — the five-field input form posting to /predict."),

        // ---------------- 6. Model training ----------------
        h1("6. Model Training Pipeline"),
        body("model_training/train_model.py is a self-contained script that:"),
        bullet("Generates a synthetic-but-realistic dataset of 2,000 samples for the five features (no original dataset was supplied with this brief — swap in your own historical CSV to retrain on real data)."),
        bullet("Splits the data 80/20 into train and test sets, stratified on the target."),
        bullet("Fits a StandardScaler on the training features."),
        bullet("Trains a GradientBoostingClassifier (200 estimators) — used here as a drop-in stand-in for XGBoost, which was unavailable in this offline environment; swap in xgboost.XGBClassifier with no other pipeline changes if you have it installed."),
        bullet("Evaluates accuracy, ROC AUC, a confusion matrix, and per-class precision/recall."),
        bullet("Persists the model and scaler to floods.save / transform.save with joblib for app.py to load."),
        body("To retrain on your own data, replace the synthetic-generation block with pd.read_csv('your_data.csv') using the same five feature columns and a FLOOD target column, then re-run the script."),

        // ---------------- 7. Setup ----------------
        h1("7. Setup & Run Instructions"),
        h2("1. Install dependencies"),
        code("cd floods_prediction"),
        code("pip install -r requirements.txt"),
        h2("2. (Optional) Retrain the model"),
        code("cd model_training"),
        code("python train_model.py"),
        body("This regenerates floods.save and transform.save in the project root. Skip this step to use the model already included with this deliverable."),
        h2("3. Run the server"),
        code("cd floods_prediction"),
        code("python app.py"),
        body("The terminal will print a local URL, typically http://127.0.0.1:5000/. Open it in a browser."),
        h2("4. Use the app"),
        bullet("Click \"Predict Floods\" on the home page to open the input form."),
        bullet("Enter cloud cover and the four rainfall figures, then click Predict."),
        bullet("You are routed to chance.html (flood risk) or no_chance.html (safe) with a confidence percentage."),

        // ---------------- 8. Results ----------------
        h1("8. Results"),
        body("On the held-out 20% test split (400 synthetic samples), the trained GradientBoostingClassifier achieved:"),
        twoColTable([
          ["Metric", "Value"],
          ["Accuracy", "82.8%"],
          ["ROC AUC", "0.871"],
          ["Precision (Flood class)", "0.85"],
          ["Recall (Flood class)", "0.93"],
        ]),
        centeredImage(img("screenshots/model_results.png"), 460, 158),
        caption("Confusion matrix and feature importance on the held-out test split."),
        body("Annual rainfall and June–September (monsoon) rainfall are the two strongest predictors, consistent with domain expectation — monsoon-season totals dominate basin saturation and runoff."),
        h2("Sample prediction — high rainfall input"),
        centeredImage(img("screenshots/chance.png"), 400, 303),
        caption("Cloud cover 80%, annual rainfall 1800mm, Jun-Sep 1200mm → predicted FLOOD (99.1% confidence)."),
        h2("Sample prediction — low rainfall input"),
        centeredImage(img("screenshots/no_chance.png"), 400, 303),
        caption("Cloud cover 10%, annual rainfall 200mm, Jun-Sep 50mm → predicted NO FLOOD."),

        // ---------------- 9. Conclusion ----------------
        h1("9. Conclusion & Next Steps"),
        body("This deliverable packages a complete, runnable flood-prediction web application: a trained classifier, a Flask backend, and a styled four-page frontend, along with the training script and evaluation results used to produce the model."),
        body("Suggested next steps for production use:"),
        bullet("Replace the synthetic training data with real historical rainfall/flood records for your region."),
        bullet("If available, swap GradientBoostingClassifier for XGBoost/XGBClassifier for a closer match to the original design."),
        bullet("Add input logging and monitoring so live predictions can be reviewed against actual outcomes over time."),
        bullet("Deploy behind a production WSGI server (e.g. gunicorn) rather than the Flask development server."),

        new Paragraph({ spacing: { before: 400 },
          children: [new TextRun({ text: "References: ", bold: true }),
            new TextRun({ text: "w3schools.com/html, w3schools.com/css, w3schools.com/js — used as general HTML/CSS/JS reference during frontend development.", color: MUTED })] }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(path.join(ROOT, "Floods_Prediction_Documentation.docx"), buffer);
  console.log("Document written.");
});
