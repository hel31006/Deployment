import pymysql
from datetime import datetime
from thefuzz import fuzz
import os # Import the 'os' module to access environment variables

def get_connection():
    """
    Establishes and returns a connection to the MySQL database.
    It reads connection details from environment variables for deployment,
    with fallback default values for local development.
    """
    # Read database credentials from environment variables.
    # The second argument is the default value, used if the environment variable is not set.
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', 'root')
    db_password = os.environ.get('DB_PASSWORD', 'zoetisproject123!')
    db_name = os.environ.get('DB_NAME', 'voice_records')

    return pymysql.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )

def get_sales_rep_id(rep_name, connection):
    """
    Finds the ID for a sales rep by name.
    If the rep does not exist, it creates a new one.
    """
    with connection.cursor() as cursor:
        cursor.execute("SELECT Sales_Rep_ID FROM sales_rep WHERE Rep_Name = %s", (rep_name,))
        result = cursor.fetchone()
        if result:
            return result["Sales_Rep_ID"]

        cursor.execute("SELECT Sales_Rep_ID FROM sales_rep ORDER BY Sales_Rep_ID DESC LIMIT 1")
        last = cursor.fetchone()
        if last and last["Sales_Rep_ID"].startswith("SR"):
            new_num = int(last["Sales_Rep_ID"][2:]) + 1
        else:
            new_num = 1
        new_id = f"SR{str(new_num).zfill(3)}"

        cursor.execute("INSERT INTO sales_rep (Sales_Rep_ID, Rep_Name) VALUES (%s, %s)", (new_id, rep_name))
        return new_id

def get_product_id(product_name, connection):
    """
    Finds the ID for a product by its exact name. Returns None if not found.
    """
    if not product_name:
        return None
    with connection.cursor() as cursor:
        cursor.execute("SELECT Product_ID FROM product WHERE LOWER(Product_Name) = %s", (product_name.lower(),))
        result = cursor.fetchone()
        if result:
            return result["Product_ID"]
        else:
            return None

def fuzzy_match_product(product_name_partial, connection, threshold=50):
    """
    Finds the best product match using fuzzy string matching.
    """
    with connection.cursor() as cursor:
        cursor.execute("SELECT Product_ID, Product_Name FROM product")
        all_products = cursor.fetchall()

    best_match = None
    best_score = 0

    for product in all_products:
        score = fuzz.partial_ratio(product_name_partial.lower(), product["Product_Name"].lower())
        if score > best_score:
            best_score = score
            best_match = product

    if best_match and best_score >= threshold:
        best_match["match_score"] = best_score
        return best_match
    else:
        return None

def match_clinic(clinic_name_partial, connection):
    """
    Performs a simple search for a clinic using SQL LIKE.
    """
    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT Clinic_ID, Clinic_Name
                FROM clinic
                WHERE Clinic_Name LIKE %s
                LIMIT 1
            """
            cursor.execute(sql, (f"%{clinic_name_partial}%",))
            return cursor.fetchone()
    except Exception as e:
        print(f"Error during exact clinic match: {e}")
        return None

def fuzzy_match_clinic(clinic_name_partial, connection, threshold=75):
    """
    Finds the best clinic match using fuzzy string matching.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT Clinic_ID, Clinic_Name FROM clinic")
            all_clinics = cursor.fetchall()

        best_match = None
        best_score = 0

        for clinic in all_clinics:
            score = fuzz.partial_ratio(clinic_name_partial.lower(), clinic["Clinic_Name"].lower())
            if score > best_score:
                best_score = score
                best_match = clinic

        if best_score >= threshold:
            best_match["match_score"] = best_score
            return best_match
        else:
            return None
    except Exception as e:
        print(f"Fuzzy match failed for clinic: {e}")
        return None

def insert_interaction(clinic_id, extracted, connection):
    """
    Inserts a new CRM interaction record into the database.
    """
    try:
        with connection.cursor() as cursor:
            sales_rep_id = get_sales_rep_id(extracted.get("Rep_Name", ""), connection)
            product_id = get_product_id(extracted.get("Product_Interest", ""), connection)

            if not product_id:
                print(f"Product not found: {extracted.get('Product_Interest', '')}, skipping insertion.")
                return None, None

            crm_created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            interaction_date = extracted.get("Interaction_Date") or crm_created_date.split(" ")[0]

            last_contacted = extracted.get("Last_Contacted")
            if not last_contacted:
                last_contacted = interaction_date

            contact_name = extracted.get("Contact_Name", "").strip()
            if not contact_name:
                cursor.execute("SELECT Contact_Name FROM clinic WHERE Clinic_ID = %s", (clinic_id,))
                row = cursor.fetchone()
                contact_name = row["Contact_Name"] if row else ""

            cursor.execute("""
                SELECT COUNT(*) as cnt FROM crm_interaction
                WHERE Clinic_ID = %s AND Sales_Rep_ID = %s AND Product_ID = %s AND DATE(Interaction_Date) = %s
            """, (clinic_id, sales_rep_id, product_id, interaction_date))

            if cursor.fetchone()["cnt"] > 0:
                print(f"Skipping duplicate interaction for clinic: {clinic_id}")
                return interaction_date, crm_created_date

            sql = """
                INSERT INTO crm_interaction (
                    Clinic_ID, Contact_Name, Sales_Rep_ID,
                    Product_ID, Samples_Given, Follow_Up,
                    Status, Interaction_Date, Additional_Notes,
                    CRM_Created_Date, Lead_Source, Last_Contacted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                clinic_id,
                contact_name,
                sales_rep_id,
                product_id,
                extracted.get("Samples_Given", ""),
                extracted.get("Follow_Up", ""),
                extracted.get("Status", ""),
                interaction_date,
                extracted.get("Additional_Notes", ""),
                crm_created_date,
                extracted.get("Lead_Source", ""),
                last_contacted
            ))

        print(f"Insert Successful for clinic: {clinic_id}")
        return interaction_date, crm_created_date

    except Exception as e:
        print(f"Error during interaction insertion: {e}")
        connection.rollback()
        return None, None