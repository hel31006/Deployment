# Import the PostgreSQL driver 'psycopg2' instead of 'pymysql'.
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
from thefuzz import fuzz
import os


def get_connection():
    """
    Establishes a connection to the PostgreSQL database using the DATABASE_URL
    environment variable provided by Render.
    """
    # Render provides the DATABASE_URL environment variable automatically.
    database_url = os.environ.get('DATABASE_URL')

    # For local testing, you would need to set this environment variable manually.
    if not database_url:
        raise Exception("DATABASE_URL environment variable is not set.")

    # The cursor_factory=DictCursor allows accessing query results by column name.
    return psycopg2.connect(
        database_url,
        cursor_factory=DictCursor
    )


def get_sales_rep_id(rep_name, connection):
    """
    Finds the ID for a sales rep by name using PostgreSQL syntax.
    If the rep does not exist, it creates a new one.
    """
    with connection.cursor() as cursor:
        # Use double quotes for identifiers to ensure compatibility.
        cursor.execute('SELECT "Sales_Rep_ID" FROM "sales_rep" WHERE "Rep_Name" = %s', (rep_name,))
        result = cursor.fetchone()
        if result:
            return result["Sales_Rep_ID"]

        cursor.execute('SELECT "Sales_Rep_ID" FROM "sales_rep" ORDER BY "Sales_Rep_ID" DESC LIMIT 1')
        last = cursor.fetchone()
        if last and last["Sales_Rep_ID"].startswith("SR"):
            new_num = int(last["Sales_Rep_ID"][2:]) + 1
        else:
            new_num = 1
        new_id = f"SR{str(new_num).zfill(3)}"

        cursor.execute('INSERT INTO "sales_rep" ("Sales_Rep_ID", "Rep_Name") VALUES (%s, %s)', (new_id, rep_name))
        return new_id


def get_product_id(product_name, connection):
    """Finds the ID for a product by its name (case-insensitive)."""
    if not product_name:
        return None
    with connection.cursor() as cursor:
        cursor.execute('SELECT "Product_ID" FROM "product" WHERE LOWER("Product_Name") = LOWER(%s)', (product_name,))
        result = cursor.fetchone()
        return result["Product_ID"] if result else None


def fuzzy_match_product(product_name_partial, connection, threshold=50):
    """Finds the best product match using fuzzy string matching."""
    with connection.cursor() as cursor:
        cursor.execute('SELECT "Product_ID", "Product_Name" FROM "product"')
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
    """Performs a case-insensitive search for a clinic using PostgreSQL's ILIKE."""
    try:
        with connection.cursor() as cursor:
            # Use ILIKE for case-insensitive matching in PostgreSQL.
            sql = 'SELECT "Clinic_ID", "Clinic_Name" FROM "clinic" WHERE "Clinic_Name" ILIKE %s LIMIT 1'
            cursor.execute(sql, (f"%{clinic_name_partial}%",))
            return cursor.fetchone()
    except Exception as e:
        print(f"Error during exact clinic match: {e}")
        return None


def fuzzy_match_clinic(clinic_name_partial, connection, threshold=75):
    """Finds the best clinic match using fuzzy string matching."""
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT "Clinic_ID", "Clinic_Name" FROM "clinic"')
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
    """Inserts a new CRM interaction record into the PostgreSQL database."""
    try:
        with connection.cursor() as cursor:
            sales_rep_id = get_sales_rep_id(extracted.get("Rep_Name", ""), connection)
            product_id = get_product_id(extracted.get("Product_Interest", ""), connection)

            if not product_id:
                print(f"Product not found: {extracted.get('Product_Interest', '')}, skipping insertion.")
                return None, None

            crm_created_date = datetime.now()
            interaction_date = extracted.get("Interaction_Date") or crm_created_date.date()
            last_contacted = extracted.get("Last_Contacted") or interaction_date
            contact_name = extracted.get("Contact_Name", "").strip()

            if not contact_name:
                cursor.execute('SELECT "Contact_Name" FROM "clinic" WHERE "Clinic_ID" = %s', (clinic_id,))
                row = cursor.fetchone()
                contact_name = row["Contact_Name"] if row else ""

            # Check for duplicates using PostgreSQL's date casting.
            cursor.execute(
                'SELECT COUNT(*) as cnt FROM "crm_interaction" WHERE "Clinic_ID" = %s AND "Sales_Rep_ID" = %s AND "Product_ID" = %s AND "Interaction_Date"::date = %s::date',
                (clinic_id, sales_rep_id, product_id, interaction_date)
            )
            if cursor.fetchone()["cnt"] > 0:
                print(f"Skipping duplicate interaction: {clinic_id}, {sales_rep_id}")
                return interaction_date, crm_created_date

            # Insert the new record.
            sql = """
                INSERT INTO "crm_interaction" (
                    "Clinic_ID", "Contact_Name", "Sales_Rep_ID", "Product_ID", "Samples_Given", "Follow_Up",
                    "Status", "Interaction_Date", "Additional_Notes", "CRM_Created_Date", "Lead_Source", "Last_Contacted"
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                clinic_id, contact_name, sales_rep_id, product_id,
                extracted.get("Samples_Given", ""), extracted.get("Follow_Up", ""),
                extracted.get("Status", ""), interaction_date, extracted.get("Additional_Notes", ""),
                crm_created_date, extracted.get("Lead_Source", ""), last_contacted
            ))
            # Note: A connection.commit() is not needed here because the 'with' block in app.py will handle it.

        print(f"Insert successful for clinic: {clinic_id}")
        return interaction_date, crm_created_date
    except Exception as e:
        print(f"Error during interaction insertion: {e}")
        # Rollback the transaction in case of an error to maintain data integrity.
        connection.rollback()
        return None, None