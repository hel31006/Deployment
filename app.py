from flask import Flask, request, redirect, url_for, session, render_template, flash, send_file, jsonify
import os, csv, io, json, zipfile, re
import whisper
from transformers import pipeline
import imageio_ffmpeg
from db_utils import get_connection, insert_interaction, get_sales_rep_id, get_product_id, match_clinic, \
    fuzzy_match_clinic
from datetime import datetime
from rapidfuzz import fuzz

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

# --- FFMPEG and Model Initialization ---
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ['PATH'] += os.pathsep + os.path.dirname(ffmpeg_path)

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Loading NER model...")
ner_model = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
print("Models loaded successfully.")

# --- Unchanged Functions ---
product_keywords = [
    "canine vaccines", "dental cleaning kits", "deworming tablets",
    "diagnostic equipment", "feline vaccines", "flea & tick prevention kits",
    "joint support supplements", "pain relief medication", "post-surgery antibiotics"
]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_clinic_name_from_text(note):
    negation_keywords = ["wait", "no", "not", "isn't", "wasn't", "aren't", "don't", "didn't", "not from", "it's not"]
    ner_results = ner_model(note)
    org_entities = [{"word": ent["word"], "start": ent.get("start", -1)} for ent in ner_results if
                    ent["entity_group"] == "ORG"]
    note_lower = note.lower()
    neg_index = -1
    for word in negation_keywords:
        idx = note_lower.find(word)
        if idx != -1:
            neg_index = idx
            break
    if org_entities:
        if neg_index != -1:
            candidates_after_neg = [ent["word"] for ent in org_entities if ent["start"] > neg_index]
            if candidates_after_neg: return candidates_after_neg[0].strip()
        return org_entities[0]["word"].strip()
    return ""


def extract_samples_and_followup(note):
    follow_up_keywords = ["follow up", "follow-up", "reconnect", "call again", "get back", "schedule another",
                          "next visit"]
    samples_keywords = ["sample", "samples", "provided", "left some", "leave", "delivered", "they received", "they got",
                        "drop off", "give them"]
    negation_pattern = re.compile(
        r"\b(no|not|didn’t|didn't|never|without|forgot|ran out|didn’t end up|was going to but didn’t|meant to but didn’t)\b",
        re.IGNORECASE)
    samples_given, follow_up = "Unknown", "Unknown"
    sentences = list(reversed(re.split(r'[.!?\n]', note)))
    for sentence in sentences:
        sentence = sentence.strip().lower()
        if not sentence: continue
        if any(kw in sentence for kw in samples_keywords):
            if negation_pattern.search(sentence):
                samples_given = "No"
            else:
                samples_given = "Yes"
            break
    for sentence in sentences:
        sentence = sentence.strip().lower()
        if not sentence: continue
        if any(kw in sentence for kw in follow_up_keywords):
            if negation_pattern.search(sentence):
                follow_up = "No"
            else:
                follow_up = "Yes"
            break
    return samples_given.title(), follow_up.title()


def match_product_from_note(note, product_keywords, threshold=70):
    note_lower = note.lower()
    best_match, best_score = None, 0
    for kw in product_keywords:
        score = fuzz.partial_ratio(kw.lower(), note_lower)
        if score > best_score and score >= threshold:
            best_score, best_match = score, kw
    return best_match


def match_sales_rep(rep_name_candidate):
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT Rep_Name FROM sales_rep")
            reps = [row['Rep_Name'] for row in cursor.fetchall()]
        best_match, best_score = None, 0
        for rep in reps:
            score = fuzz.ratio(rep.lower(), rep_name_candidate.lower())
            if score > best_score and score >= 45:
                best_score, best_match = score, rep
        return best_match
    finally:
        connection.close()


def extract_fields(note):
    ner_results = ner_model(note)
    rep_name, contact_name, clinic_name, product_interest = "", "", "", ""
    per_entities = [ent['word'] for ent in ner_results if ent['entity_group'] == 'PER']
    if per_entities:
        rep_name = per_entities[0]
        if len(per_entities) > 1: contact_name = per_entities[1]
    product_interest = match_product_from_note(note, product_keywords)
    clinic_name = extract_clinic_name_from_text(note)
    samples_given, follow_up = extract_samples_and_followup(note)
    matched_rep = match_sales_rep(rep_name)
    if matched_rep: rep_name = matched_rep
    status = "Unknown"
    status_keywords = {"closed - converted": ["closed and converted", "successfully closed"],
                       "closed - not converted": ["closed but not converted", "not converted"],
                       "working": ["still working", "currently working", "in progress"],
                       "new": ["new lead", "first time"]}
    for label, keywords in status_keywords.items():
        if status != "Unknown": break
        for phrase in keywords:
            if phrase in note.lower():
                status = label
                break
    lead_source = "Unknown"
    lead_source_keywords = {"web form": "Web Form", "referral": "Referral", "phone": "Phone Inquiry",
                            "call": "Phone Inquiry", "trade show": "Trade Show", "email": "Email Campaign"}
    for keyword, label in lead_source_keywords.items():
        if keyword in note.lower():
            lead_source = label
            break
    return {"Rep_Name": rep_name, "Contact_Name": contact_name, "Clinic_Name": clinic_name,
            "Product_Interest": product_interest, "Samples_Given": samples_given, "Follow_Up": follow_up,
            "Status": status.title(), "Lead_Source": lead_source}


def clean_clinic_name(name):
    prefixes = ['out of', 'at', 'from', 'to', 'of', 'in', 'inside', 'near', 'beside']
    name = name.strip().lower()
    for prefix in prefixes:
        if name.startswith(prefix + " "):
            name = name[len(prefix) + 1:]
            break
    return name.strip().title()


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            flash('No selected file')
            return redirect(request.url)
        old_clients, new_clients = [], []
        for file in files:
            if file and allowed_file(file.filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                result = whisper_model.transcribe(filepath)
                transcription = result["text"]
                extracted = extract_fields(transcription)
                extracted["filename"] = file.filename
                extracted["transcription"] = transcription
                cleaned_name = clean_clinic_name(extracted.get("Clinic_Name", ""))
                extracted["Clinic_Name"] = cleaned_name
                with get_connection() as connection:
                    clinic_info = match_clinic(cleaned_name, connection)
                    if clinic_info:
                        extracted.update({"clinic_id": clinic_info["Clinic_ID"], "match_type": "exact"})
                        old_clients.append(extracted)
                    else:
                        fuzzy_result = fuzzy_match_clinic(cleaned_name, connection)
                        if fuzzy_result:
                            extracted.update({"clinic_id": fuzzy_result["Clinic_ID"],
                                              "clinic_name_matched": fuzzy_result["Clinic_Name"],
                                              "match_score": fuzzy_result["match_score"], "match_type": "fuzzy"})
                            old_clients.append(extracted)
                        else:
                            extracted["match_type"] = "new"
                            new_clients.append(extracted)
        return render_template("review.html", old_clients=old_clients, new_clients=new_clients)
    return render_template("index.html")


@app.route('/submit_existing', methods=['POST'])
def submit_existing():
    count = int(request.form.get("count", 0))
    new_clients_from_upload = json.loads(request.form.get("new_clients_json", "[]"))

    existing_interactions, confirmed_new_clients = [], []
    processed_keys = set()

    with get_connection() as connection:
        for i in range(count):
            decision = request.form.get(f"clinic_decision_{i}", "existing")

            record_data = {
                "clinic_id": request.form.get(f"clinic_id_{i}"),
                "Clinic_Name": request.form.get(f"clinic_name_{i}", "").strip(),
                "Contact_Name": request.form.get(f"contact_name_{i}", "").strip(),
                "Rep_Name": request.form.get(f"rep_name_{i}", "").strip(),
                "Product_Interest": request.form.get(f"product_interest_{i}", "").strip(),
                "Samples_Given": request.form.get(f"samples_given_{i}", "").strip(),
                "Follow_Up": request.form.get(f"follow_up_{i}", "").strip(),
                "Status": request.form.get(f"status_{i}", "").strip(),
                "Lead_Source": request.form.get(f"lead_source_{i}", "").strip(),
                "Last_Contacted": request.form.get(f"last_contacted_{i}", "").strip(),
                "Additional_Notes": request.form.get(f"additional_notes_{i}", "").strip(),
                "transcription": request.form.get(f"transcription_{i}", "").strip(),
                "filename": request.form.get(f"filename_{i}", "").strip()
            }

            if decision == "new":
                confirmed_new_clients.append(record_data)
                continue

            clinic_id = record_data["clinic_id"]
            if not clinic_id:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT Clinic_ID FROM clinic WHERE LOWER(TRIM(Clinic_Name)) = LOWER(TRIM(%s))",
                                   (record_data["Clinic_Name"],))
                    row = cursor.fetchone()
                    clinic_id = row["Clinic_ID"] if row else None
            if not clinic_id: continue

            interaction_key = (clinic_id, record_data["Product_Interest"].lower(), record_data["Rep_Name"].lower())
            if interaction_key in processed_keys: continue

            interaction_date, crm_created_date = insert_interaction(clinic_id, record_data, connection)
            processed_keys.add(interaction_key)

            existing_interactions.append({
                "Clinic_ID": clinic_id, "Contact_Name": record_data["Contact_Name"],
                "Rep_Name": record_data["Rep_Name"],
                "Product_Interest": record_data["Product_Interest"], "Samples_Given": record_data["Samples_Given"],
                "Follow_Up": record_data["Follow_Up"], "Status": record_data["Status"],
                "Interaction_Date": interaction_date,
                "Lead_Source": record_data["Lead_Source"], "Last_Contacted": record_data["Last_Contacted"],
                "Additional_Notes": record_data["Additional_Notes"], "CRM_Created_Date": crm_created_date,
                "Transcription": record_data["transcription"], "Filename": record_data["filename"]
            })
        connection.commit()

    final_new_clients = new_clients_from_upload + confirmed_new_clients

    seen_filenames = set()
    unique_new_clients = []
    for client in final_new_clients:
        if client.get('filename') and client['filename'] not in seen_filenames:
            unique_new_clients.append(client)
            seen_filenames.add(client['filename'])
        elif not client.get('filename'):
             unique_new_clients.append(client)

    session["submitted_interactions"] = session.get("submitted_interactions", []) + existing_interactions
    flash("Existing clinic interactions processed successfully.")
    return render_template("bulk_new_clinic_form.html", new_clients=unique_new_clients)


# --- Other routes are unchanged ---
@app.route("/submit_new_clinics", methods=["POST"])
def submit_new_clinics():
    count = int(request.form.get("count", 0))
    new_interaction_records = []
    with get_connection() as connection:
        for i in range(count):
            clinic_name = request.form.get(f"clinic_name_{i}")
            clinic_type = request.form.get(f"clinic_type_{i}")
            industry = request.form.get(f"industry_{i}")
            address = request.form.get(f"address_{i}")
            region = request.form.get(f"region_{i}")
            parent_company = request.form.get(f"parent_company_{i}")
            contact_name = request.form.get(f"contact_name_{i}")
            rep_name = request.form.get(f"rep_name_{i}")
            product_name = request.form.get(f"product_interest_{i}")
            interaction_date = request.form.get(f"interaction_date_{i}") or datetime.now().strftime("%Y-%m-%d")
            follow_up = request.form.get(f"follow_up_{i}")
            samples_given = request.form.get(f"samples_given_{i}")
            status = request.form.get(f"status_{i}")
            lead_source = request.form.get(f"lead_source_{i}")
            last_contacted = request.form.get(f"last_contacted_{i}") or interaction_date
            additional_notes = request.form.get(f"additional_notes_{i}")
            transcription = request.form.get(f"transcription_{i}", "")
            filename = request.form.get(f"filename_{i}", "")
            with connection.cursor() as cursor:
                cursor.execute("SELECT MAX(Clinic_ID) AS max_id FROM clinic WHERE Clinic_ID REGEXP '^C[0-9]+$'")
                row = cursor.fetchone()
                new_id_num = int(row["max_id"][1:]) + 1 if row and row["max_id"] else 1
                clinic_id = f"C{str(new_id_num).zfill(3)}"
                cursor.execute(
                    "INSERT INTO clinic (Clinic_ID, Clinic_Name, Clinic_Type, Industry, Clinic_Address, Region, Parent_Company, Contact_Name) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (clinic_id, clinic_name, clinic_type, industry, address, region, parent_company, contact_name))
                sales_rep_id = get_sales_rep_id(rep_name, connection)
                product_id = get_product_id(product_name, connection)
                crm_created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    "INSERT INTO crm_interaction (Clinic_ID, Contact_Name, Sales_Rep_ID, Product_ID, Interaction_Date, Follow_Up, Samples_Given, Status, Lead_Source, Last_Contacted, Additional_Notes, CRM_Created_Date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (clinic_id, contact_name, sales_rep_id, product_id, interaction_date, follow_up, samples_given,
                     status, lead_source, last_contacted, additional_notes, crm_created_date))
                new_interaction_records.append(
                    {"Clinic_ID": clinic_id, "Contact_Name": contact_name, "Rep_Name": rep_name,
                     "Product_Interest": product_name, "Interaction_Date": interaction_date, "Follow_Up": follow_up,
                     "Samples_Given": samples_given, "Status": status, "Lead_Source": lead_source,
                     "Last_Contacted": last_contacted, "Additional_Notes": additional_notes,
                     "CRM_Created_Date": crm_created_date, "Transcription": transcription, "Filename": filename})
        connection.commit()
    session["submitted_interactions"] = session.get("submitted_interactions", []) + new_interaction_records
    return redirect(url_for("submission_success"))


@app.route("/get_product_list")
def get_product_list():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT Product_Name FROM product")
            return jsonify([row["Product_Name"] for row in cursor.fetchall()])


@app.route("/get_clinic_list")
def get_clinic_list():
    keyword = request.args.get("q", "").strip().lower()
    if not keyword: return jsonify([])
    with get_connection() as connection:
        with connection.cursor() as cursor:
            query = "SELECT Clinic_ID, Clinic_Name FROM clinic WHERE LOWER(Clinic_Name) LIKE %s LIMIT 10"
            cursor.execute(query, (f"%{keyword}%",))
            return jsonify(
                [{"Clinic_ID": row["Clinic_ID"], "Clinic_Name": row["Clinic_Name"]} for row in cursor.fetchall()])


@app.route("/get_rep_list")
def get_rep_list():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT Rep_Name FROM sales_rep ORDER BY Rep_Name ASC")
            return jsonify([row["Rep_Name"] for row in cursor.fetchall() if row.get("Rep_Name")])


@app.route("/submission_success")
def submission_success():
    return render_template("submission_success.html")


@app.route("/download_csvs")
def download_csvs():
    interactions = session.get("submitted_interactions", [])
    if not interactions: return "No CRM interaction data available.", 400
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        crm_buffer = io.StringIO()
        fieldnames = ["Clinic_ID", "Contact_Name", "Rep_Name", "Product_Interest", "Samples_Given", "Follow_Up",
                      "Status", "Interaction_Date", "Lead_Source", "Last_Contacted", "Additional_Notes",
                      "CRM_Created_Date", "Transcription", "Filename"]
        writer = csv.DictWriter(crm_buffer, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(interactions)
        zipf.writestr("crm_interaction.csv", crm_buffer.getvalue())
    session.pop("submitted_interactions", None)
    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name="crm_interactions.zip")


# if __name__ == '__main__':
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])
#     app.run(debug=True)