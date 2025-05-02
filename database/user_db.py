from .models import get_connection
import bcrypt

def create_user(fullname, gender, contact, email, password):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        query = "INSERT INTO users (fullname, gender, contact, email, password) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(query, (fullname, gender, contact, email, hashed))
        conn.commit()
        print("Inserted successfully:", fullname)
        return True
    except Exception as e:
        print("Error inserting user:", e)
        return False
    finally:
        conn.close()



def verify_user(email, password):
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT password FROM users WHERE email = %s"
    cursor.execute(query, (email,))
    result = cursor.fetchone()
    conn.close()
    if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
        return True
    return False
