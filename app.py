# ✅ app.py 最终部署版

from flask import Flask, request, redirect, url_for, session, render_template, flash, send_file, jsonify
import os, csv, io, json, zipfile, re
import whisper
from transformers import pipeline
import imageio_ffmpeg
from db_utils import get_connection, insert_interaction, get_sales_rep_id, get_product_id, match_clinic, fuzzy_match_clinic
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ['PATH'] += os.pathsep + os.path.dirname(ffmpeg_path)

whisper_model = whisper.load_model("base")
ner_model = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

product_keywords = [
    "canine vaccines", "dental cleaning kits", "deworming tablets",
    "diagnostic equipment", "feline vaccines", "flea & tick prevention kits",
    "joint support supplements", "pain relief medication", "post-surgery antibiotics"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_clinic_name_from_text(note):
    pattern = r"(?:\\bat\\b|\\bfrom\\b|\\bover at\\b|\\bvisit(?:ed)?\\b|\\bgot out of\\b|\\bleft\\b|\\bstepped out of\\b)\\s+([A-Z][\\w&', \-]+(Clinic|Hospital|Care|Vet|Group|Center|Ltd))"
    match = re.search(pattern, note)
    return match.group(1).strip() if match else ""

def extract_fields(note):
    ner_results = ner_model(note)
    rep_name, contact_name, clinic_name, product_interest = "", "", "", ""

    per_entities = [ent['word'] for ent in ner_results if ent['entity_group'] == 'PER']
    org_entities = [ent['word'] for ent in ner_results if ent['entity_group'] == 'ORG']
    if per_entities:
        rep_name = per_entities[0]
        if len(per_entities) > 1:
            contact_name = per_entities[1]
    if org_entities:
        clinic_name = org_entities[0]
    if not clinic_name:
        clinic_name = extract_clinic_name_from_text(note)

    for kw in product_keywords:
        if kw.lower() in note.lower():
            product_interest = kw
            break

    negation_pattern = re.compile(r"\\b(no|not|didn’t|didn't|never|without)\\b", re.IGNORECASE)
    samples_given, follow_up = "unknown", "unknown"
    for sentence in re.split(r'[.!?]', note):
        if "sample" in sentence.lower():
            samples_given = "no" if negation_pattern.search(sentence) else "yes"
        if any(x in sentence.lower() for x in ["follow up", "follow-up", "following up"]):
            follow_up = "no" if negation_pattern.search(sentence) else "yes"

    note_lower = note.lower()
    if "closed - converted" in note_lower:
        status = "closed - converted"
    elif "closed - not converted" in note_lower:
        status = "closed - not converted"
    elif "working" in note_lower:
        status = "working"
    elif "new" in note_lower:
        status = "new"
    else:
        status = "unknown"

    return {
        "Rep_Name": rep_name,
        "Contact_Name": contact_name,
        "Clinic_Name": clinic_name,
        "Product_Interest": product_interest,
        "Samples_Given": samples_given,
        "Follow_Up": follow_up,
        "Status": status
    }

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('files')
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
                extracted["Lead_Source"] = ""

                if not extracted.get("Clinic_Name"):
                    continue

                with get_connection() as connection:
                    clinic_info = match_clinic(extracted.get("Clinic_Name"), connection)
                    if clinic_info:
                        extracted.update({
                            "clinic_id": clinic_info["Clinic_ID"],
                            "clinic_name_matched": clinic_info["Clinic_Name"],
                            "match_type": "exact"
                        })
                        old_clients.append(extracted)
                    else:
                        fuzzy_result = fuzzy_match_clinic(extracted.get("Clinic_Name"), connection)
                        if fuzzy_result:
                            extracted.update({
                                "clinic_id": fuzzy_result["Clinic_ID"],
                                "clinic_name_matched": fuzzy_result["Clinic_Name"],
                                "match_score": fuzzy_result["match_score"],
                                "match_type": "fuzzy"
                            })
                            old_clients.append(extracted)
                        else:
                            extracted["match_type"] = "new"
                            new_clients.append(extracted)

        return render_template("review.html", old_clients=old_clients, new_clients=new_clients)

    return render_template("index.html")

@app.route("/get_product_list")
def get_product_list():
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT Product_Name FROM product")
            products = cursor.fetchall()
            product_names = [row["Product_Name"] for row in products]
    finally:
        connection.close()

    return jsonify(product_names)

@app.route('/submit_existing', methods=['POST'])
def submit_existing():
    count = int(request.form.get("count", 0))
    new_clients_json = request.form.get("new_clients_json")
    new_clients_all = json.loads(new_clients_json or "[]")
    existing_clinics, existing_interactions = [], []
    confirmed_new_clients = []
    processed_keys = set()

    connection = get_connection()
    try:
        for i in range(count):
            decision = request.form.get(f"clinic_decision_{i}", "existing")
            clinic_id = request.form.get(f"clinic_id_{i}")
            clinic_name = request.form.get(f"clinic_name_{i}")
            rep_name = request.form.get(f"rep_name_{i}")
            product_interest = request.form.get(f"product_interest_{i}")
            samples_given = request.form.get(f"samples_given_{i}")
            follow_up = request.form.get(f"follow_up_{i}")
            status = request.form.get(f"status_{i}")
            additional_notes = request.form.get(f"additional_notes_{i}")
            lead_source = request.form.get(f"lead_source_{i}")
            last_contacted = request.form.get(f"last_contacted_{i}") or None

            interaction_key = (clinic_id, product_interest.strip().lower(), rep_name.strip().lower())

            if interaction_key not in processed_keys:
                interaction_date, crm_created_date = insert_interaction(clinic_id, {
                    "Rep_Name": rep_name,
                    "Product_Interest": product_interest,
                    "Samples_Given": samples_given,
                    "Follow_Up": follow_up,
                    "Status": status,
                    "Additional_Notes": additional_notes,
                    "Lead_Source": lead_source,
                    "Last_Contacted": last_contacted
                }, connection)

                with connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM clinic WHERE Clinic_ID = %s", (clinic_id,))
                    clinic_row = cursor.fetchone()
                    if clinic_row:
                        existing_clinics.append(dict(clinic_row))


                        contact_name_from_db = clinic_row.get("Contact_Name", "")
                    else:
                        contact_name_from_db = ""

                existing_interactions.append({
                    "Clinic_ID": clinic_id,
                    "Contact_Name": contact_name_from_db,
                    "Rep_Name": rep_name,
                    "Product_Interest": product_interest,
                    "Samples_Given": samples_given,
                    "Follow_Up": follow_up,
                    "Status": status,
                    "Interaction_Date": interaction_date,
                    "Additional_Notes": additional_notes,
                    "Lead_Source": lead_source,
                    "Last_Contacted": last_contacted,
                    "CRM_Created_Date": crm_created_date
                })

                processed_keys.add(interaction_key)


        for i in range(count):
            decision = request.form.get(f"clinic_decision_{i}")
            if decision == "new" and i < len(new_clients_all):
                confirmed_new_clients.append(new_clients_all[i])


        for client in new_clients_all:
            if client.get("match_type") == "new":
                confirmed_new_clients.append(client)


        seen = set()
        unique_clients = []
        for client in confirmed_new_clients:
            cname = client.get("Clinic_Name", "").lower()
            if cname and cname not in seen:
                unique_clients.append(client)
                seen.add(cname)

        confirmed_new_clients = unique_clients

        connection.commit()
    except Exception as e:
        print("❌ Existing Clinic Upload Failed:", e)
        connection.rollback()
    finally:
        connection.close()

    session["submitted_interactions"] = session.get("submitted_interactions", []) + existing_interactions
    flash("✅ Existing Clinic Upload Successful")
    return render_template("bulk_new_clinic_form.html", new_clients=confirmed_new_clients)


@app.route("/submit_new_clinics", methods=["POST"])
def submit_new_clinics():
    count = int(request.form.get("count", 0))
    new_interaction_records = []

    connection = get_connection()
    try:
        for i in range(1, count + 1):
            clinic_name = request.form.get(f"clinic_name_{i}")
            clinic_type = request.form.get(f"clinic_type_{i}")
            industry = request.form.get(f"industry_{i}")
            address = request.form.get(f"address_{i}")
            region = request.form.get(f"region_{i}")
            parent_company = request.form.get(f"parent_company_{i}")
            contact_name = request.form.get(f"contact_name_{i}")
            rep_name = request.form.get(f"rep_name_{i}")
            product_name = request.form.get(f"product_name_{i}")
            interaction_date = request.form.get(f"interaction_date_{i}") or datetime.now().strftime("%Y-%m-%d")
            follow_up = request.form.get(f"follow_up_{i}")
            samples_given = request.form.get(f"samples_given_{i}")
            status = request.form.get(f"status_{i}")
            lead_source = request.form.get(f"lead_source_{i}")
            last_contacted = request.form.get(f"last_contacted_{i}") or interaction_date
            additional_notes = request.form.get(f"additional_notes_{i}")

            with connection.cursor() as cursor:
                cursor.execute("SELECT MAX(Clinic_ID) AS max_id FROM clinic WHERE Clinic_ID REGEXP '^C[0-9]+$'")
                row = cursor.fetchone()
                if row and row["max_id"]:
                    max_id_num = int(row["max_id"][1:])
                    new_id_num = max_id_num + 1
                else:
                    new_id_num = 1

                clinic_id = f"C{str(new_id_num).zfill(3)}"

                cursor.execute("""
                    INSERT INTO clinic (
                        Clinic_ID, Clinic_Name, Clinic_Type,
                        Industry, Clinic_Address, Region,
                        Parent_Company, Contact_Name
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    clinic_id, clinic_name, clinic_type, industry,
                    address, region, parent_company, contact_name
                ))


                sales_rep_id = get_sales_rep_id(rep_name, connection)
                product_id = get_product_id(product_name, connection)
                crm_created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


                cursor.execute("""
                    INSERT INTO crm_interaction (
                        Clinic_ID, Contact_Name, Sales_Rep_ID, Product_ID,
                        Interaction_Date, Follow_Up, Samples_Given,
                        Status, Lead_Source, Last_Contacted,
                        Additional_Notes, CRM_Created_Date
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    clinic_id, contact_name, sales_rep_id, product_id,
                    interaction_date, follow_up, samples_given,
                    status, lead_source, last_contacted,
                    additional_notes, crm_created_date
                ))


                new_interaction_records.append({
                    "Clinic_ID": clinic_id,
                    "Contact_Name": contact_name,
                    "Rep_Name": rep_name,
                    "Product_Interest": product_name,
                    "Interaction_Date": interaction_date,
                    "Follow_Up": follow_up,
                    "Samples_Given": samples_given,
                    "Status": status,
                    "Lead_Source": lead_source,
                    "Last_Contacted": last_contacted,
                    "Additional_Notes": additional_notes,
                    "CRM_Created_Date": crm_created_date
                })

        connection.commit()

    except Exception as e:
        print("❌ Insert Failed:", e)
        connection.rollback()

    finally:
        connection.close()

    session["submitted_interactions"] = session.get("submitted_interactions", []) + new_interaction_records
    return redirect(url_for("submission_success"))

@app.route("/submission_success")
def submission_success():
    return render_template("submission_success.html")

@app.route("/download_csvs")
def download_csvs():
    interactions = session.get("submitted_interactions", [])

    if not interactions:
        return "No CRM interaction data available.", 400

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        crm_buffer = io.StringIO()


        fieldnames = [
            "Clinic_ID", "Contact_Name", "Rep_Name", "Product_Interest",
            "Samples_Given", "Follow_Up", "Status", "Interaction_Date",
            "Lead_Source", "Last_Contacted", "Additional_Notes", "CRM_Created_Date"
        ]

        writer = csv.DictWriter(crm_buffer, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(interactions)

        zipf.writestr("crm_interaction.csv", crm_buffer.getvalue())

    session.pop("submitted_interactions", None)

    memory_file.seek(0)
    return send_file(memory_file,
                     mimetype='application/zip',
                     as_attachment=True,
                     download_name="crm_interactions.zip")

if __name__ == '__main__':
    app.run(debug=True)
