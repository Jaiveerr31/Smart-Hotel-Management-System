"""
Advanced Hotel Management System with Gradio UI
Features: Interactive UI, Detailed Billing, AI Sentiment Analysis, Analytics & Master Lists.
Database: SQLite (local path configurable through HOTEL_DB_PATH)
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import gradio as gr
import random
from faker import Faker
import os
import warnings

# Local database path; hosting persistence will be configured separately.
DB_PATH = os.environ.get("HOTEL_DB_PATH", "hotel_management_persistent.db")

# Import Hugging Face Transformers for free, local AI
warnings.filterwarnings('ignore')
from transformers import pipeline

# ============================================================================
# DATABASE SETUP & FAKE DATA SEEDING
# ============================================================================

class HotelDatabase:
    def __init__(self, db_name=DB_PATH):
        self.db_name = db_name
        self.fake = Faker('en_IN')
        self.init_database()

    def get_connection(self):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''CREATE TABLE IF NOT EXISTS rooms (room_id INTEGER PRIMARY KEY, room_number TEXT UNIQUE NOT NULL, room_type TEXT NOT NULL, capacity INTEGER, base_price REAL, status TEXT DEFAULT 'available', amenities TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS customers (customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE, phone TEXT, address TEXT, total_visits INTEGER DEFAULT 0, total_spent REAL DEFAULT 0, vip_status TEXT DEFAULT 'regular', preferences TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS bookings (booking_id INTEGER PRIMARY KEY, customer_id INTEGER, room_id INTEGER, check_in DATE, check_out DATE, booking_status TEXT DEFAULT 'confirmed', base_price REAL, dynamic_price REAL, discount REAL DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(customer_id) REFERENCES customers(customer_id), FOREIGN KEY(room_id) REFERENCES rooms(room_id))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS visit_history (visit_id INTEGER PRIMARY KEY, customer_id INTEGER, booking_id INTEGER, room_id INTEGER, check_in DATE, check_out DATE, nights INTEGER, spent REAL, rating REAL, feedback TEXT, sentiment_label TEXT, sentiment_score REAL, visited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(customer_id) REFERENCES customers(customer_id))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS addon_services (service_id INTEGER PRIMARY KEY, service_name TEXT UNIQUE NOT NULL, price REAL, description TEXT, category TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS booking_services (booking_service_id INTEGER PRIMARY KEY, booking_id INTEGER, service_id INTEGER, quantity INTEGER DEFAULT 1, price REAL, added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(booking_id) REFERENCES bookings(booking_id), FOREIGN KEY(service_id) REFERENCES addon_services(service_id))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS bills (bill_id INTEGER PRIMARY KEY, booking_id INTEGER, customer_id INTEGER, room_charges REAL, addon_charges REAL, late_checkout_penalty REAL DEFAULT 0, discount_amount REAL DEFAULT 0, total_amount REAL, payment_status TEXT DEFAULT 'pending', generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(booking_id) REFERENCES bookings(booking_id), FOREIGN KEY(customer_id) REFERENCES customers(customer_id))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS alerts (alert_id INTEGER PRIMARY KEY, booking_id INTEGER, alert_type TEXT, message TEXT, is_read INTEGER DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(booking_id) REFERENCES bookings(booking_id))''')

        conn.commit()
        self.seed_initial_data(conn)
        conn.close()

    def seed_initial_data(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rooms")
        if cursor.fetchone()[0] > 0:
            return

        print("🌱 Seeding database with fake guests and diverse rooms...")
        amenities_pool = ['WiFi', 'AC', 'TV', 'Minibar', 'Balcony', 'Jacuzzi', 'Ocean View', 'Work Desk', 'Kitchenette']
        rooms = []
        room_types = [('Single', 1, 3500), ('Double', 2, 5500), ('Deluxe', 2, 8500), ('Suite', 4, 15000), ('Family', 5, 12000), ('Penthouse', 4, 35000)]
        for floor in range(1, 6):
            for i in range(1, 6):
                r_type, cap, price = random.choice(room_types)
                r_num = f"{floor}0{i}"
                r_amenities = json.dumps(random.sample(amenities_pool, random.randint(3, 7)))
                rooms.append((r_num, r_type, cap, price, 'available', r_amenities))
        cursor.executemany('INSERT INTO rooms (room_number, room_type, capacity, base_price, status, amenities) VALUES (?, ?, ?, ?, ?, ?)', rooms)

        customers = []
        for _ in range(30):
            prefs = json.dumps(random.sample(amenities_pool, 3))
            customers.append((self.fake.name(), self.fake.email(), self.fake.phone_number(), self.fake.city(), prefs))
        cursor.executemany('INSERT INTO customers (name, email, phone, address, preferences) VALUES (?, ?, ?, ?, ?)', customers)

        services = [
            ('Room Service', 500.0, 'In-room dining', 'Food & Beverage'),
            ('Spa Treatment', 2000.0, 'Full body spa', 'Wellness'),
            ('Airport Pickup', 1000.0, 'Airport transportation', 'Transport'),
            ('Late Checkout', 1500.0, 'Extended checkout time', 'Room'),
            ('Laundry Service', 300.0, 'Clothes washing', 'Services'),
            ('Premium WiFi', 200.0, 'High-speed internet', 'Connectivity')
        ]
        cursor.executemany('INSERT INTO addon_services (service_name, price, description, category) VALUES (?, ?, ?, ?)', services)
        conn.commit()

# ============================================================================
# HOTEL MANAGEMENT SYSTEM
# ============================================================================

class HotelManagementSystem:
    def __init__(self):
        self.db = HotelDatabase()
        print("🤖 Loading AI Sentiment Analysis Model... (May take a moment)")
        try:
            self.nlp_sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            print("✅ AI Model Loaded Successfully!")
        except Exception as e:
            print(f"⚠️ Using fallback pipeline due to: {e}")
            self.nlp_sentiment = pipeline("sentiment-analysis")

    # ========== CUSTOMER & ROOM MANAGEMENT ==========
    def add_customer(self, name: str, email: str, phone: str, address: str) -> Tuple[bool, str]:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO customers (name, email, phone, address) VALUES (?, ?, ?, ?)', (name, email, phone, address))
            conn.commit()
            c_id = cursor.lastrowid
            conn.close()
            return True, f"Customer added with ID: {c_id}"
        except:
            conn.close()
            return False, "Error adding customer (Email might exist)"

    def get_customer_history(self, email: str) -> str:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT customer_id, name, total_visits, total_spent, vip_status FROM customers WHERE email = ?', (email,))
        customer = cursor.fetchone()
        if not customer:
            conn.close()
            return "Customer not found"

        cursor.execute('SELECT check_in, check_out, nights, spent, rating, feedback, sentiment_label FROM visit_history WHERE customer_id = ? ORDER BY visited_at DESC', (customer[0],))
        visits = cursor.fetchall()
        conn.close()

        res = f"\n{'='*60}\nCUSTOMER PROFILE\n{'='*60}\nName: {customer[1]}\nTotal Visits: {customer[2]}\nTotal Spent: ₹{customer[3]:,.2f}\nStatus: {customer[4].upper()}\n\n{'='*60}\nVISIT HISTORY\n{'='*60}\n"
        if visits:
            for v in visits: res += f"Check-in: {v[0]} | Check-out: {v[1]} | Spent: ₹{v[3]:,.2f}\nFeedback: {v[5]} [AI: {v[6]}]\n{'-'*40}\n"
        else: res += "No visit history."
        return res

    def get_available_rooms_df(self, room_type: str = "All") -> pd.DataFrame:
        conn = self.db.get_connection()
        query = "SELECT room_number as 'Room', room_type as 'Type', capacity as 'Capacity', base_price as 'Price (₹)', amenities as 'Amenities' FROM rooms WHERE status = 'available'"
        if room_type != "All": query += f" AND room_type = '{room_type}'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def recommend_rooms(self, customer_email: str, check_in: str, check_out: str, budget: float) -> str:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT customer_id, vip_status, preferences FROM customers WHERE email = ?', (customer_email,))
        c_res = cursor.fetchone()
        if not c_res:
            conn.close()
            return "Customer not found."

        vip, prefs = c_res[1], (json.loads(c_res[2]) if c_res[2] else [])
        cursor.execute("SELECT room_number, room_type, base_price, amenities FROM rooms WHERE status = 'available' AND base_price <= ?", (budget,))
        rooms = cursor.fetchall()
        conn.close()

        scored = []
        for r in rooms:
            score = 50 if vip == 'vip' else (30 if vip == 'premium' else 0)
            amens = json.loads(r[3])
            score += sum([20 for a in amens if a in prefs]) + ((budget - r[2]) / 1000)
            scored.append((score, r))
        scored.sort(reverse=True, key=lambda x: x[0])

        res = f"\n{'='*80}\nSMART RECOMMENDATIONS\n{'='*80}\n"
        for i, (score, r) in enumerate(scored[:5], 1): res += f"{i}. Room {r[0]} ({r[1]}) - ₹{r[2]:,.2f}\n   Amenities: {', '.join(json.loads(r[3]))}\n\n"
        return res

    # ========== BOOKING & PRICING ==========
    def calculate_dynamic_price(self, room_id: int, check_in: str, check_out: str, current_bookings: int) -> float:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT base_price FROM rooms WHERE room_id = ?', (room_id,))
        base = cursor.fetchone()[0]
        conn.close()

        c_in, c_out = datetime.strptime(check_in, '%Y-%m-%d'), datetime.strptime(check_out, '%Y-%m-%d')
        nights = max(1, (c_out - c_in).days)
        mult = 1.0 + (max(0, (current_bookings / 25) - 0.6) * 0.5)
        if c_in.weekday() >= 4: mult += 0.15
        return round(float(base) * mult * nights, 2)

    def create_booking(self, customer_email: str, room_number: str, check_in: str, check_out: str) -> Tuple[bool, str]:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT customer_id FROM customers WHERE email = ?', (customer_email,))
        c_id = cursor.fetchone()
        if not c_id:
            conn.close()
            return False, "Customer not found"

        cursor.execute("SELECT room_id, base_price FROM rooms WHERE room_number = ? AND status = 'available'", (room_number,))
        r_info = cursor.fetchone()
        if not r_info:
            conn.close()
            return False, "Room not available"

        cursor.execute("SELECT COUNT(*) FROM bookings WHERE booking_status = 'confirmed'")
        dyn_price = self.calculate_dynamic_price(r_info[0], check_in, check_out, cursor.fetchone()[0])

        try:
            cursor.execute('INSERT INTO bookings (customer_id, room_id, check_in, check_out, base_price, dynamic_price) VALUES (?, ?, ?, ?, ?, ?)', (c_id[0], r_info[0], check_in, check_out, r_info[1], dyn_price))
            b_id = cursor.lastrowid
            cursor.execute("UPDATE rooms SET status = 'booked' WHERE room_id = ?", (r_info[0],))
            cursor.execute("INSERT INTO alerts (booking_id, alert_type, message) VALUES (?, ?, ?)", (b_id, "New Booking", f"Room {room_number} booked for {check_in}"))
            conn.commit()
            conn.close()
            return True, f"✅ Booking ID: {b_id} created successfully! Total Price: ₹{dyn_price:,.2f}"
        except:
            conn.close()
            return False, "Booking failed"

    def get_all_bookings_df(self) -> pd.DataFrame:
        conn = self.db.get_connection()
        df = pd.read_sql_query("SELECT b.booking_id as 'ID', c.name as 'Guest', r.room_number as 'Room', b.check_in as 'Check-In', b.check_out as 'Check-Out', b.booking_status as 'Status', b.dynamic_price as 'Total (₹)' FROM bookings b JOIN customers c ON b.customer_id = c.customer_id JOIN rooms r ON b.room_id = r.room_id ORDER BY b.booking_id DESC", conn)
        conn.close()
        return df

    # ========== SERVICES & BILLING ==========
    def get_addon_services(self) -> str:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT service_name, price, description, category FROM addon_services')
        services = cursor.fetchall()
        conn.close()
        res = f"\n{'='*80}\nAVAILABLE ADD-ON SERVICES\n{'='*80}\n"
        for s in services: res += f"• [{s[3]}] {s[0]}: ₹{s[1]:,.2f} - {s[2]}\n"
        return res

    def add_services_to_booking(self, booking_id: int, service_names: List[str]) -> Tuple[bool, str]:
        if not booking_id: return False, "⚠️ Please select a valid Booking."
        if not service_names: return False, "⚠️ No services selected."

        if isinstance(service_names, str):
            service_names = [service_names]

        try:
            b_id = int(booking_id)
        except (ValueError, TypeError):
            return False, "⚠️ Invalid Booking ID format."

        conn = self.db.get_connection()
        cursor = conn.cursor()
        try:
            total_cost = 0.0
            added_list = []
            for service_name in service_names:
                cursor.execute('SELECT service_id, price FROM addon_services WHERE service_name = ?', (service_name.strip(),))
                result = cursor.fetchone()
                if result:
                    service_id, price = result
                    total_cost += float(price)
                    added_list.append(service_name)
                    cursor.execute('INSERT INTO booking_services (booking_id, service_id, quantity, price) VALUES (?, ?, 1, ?)',
                                   (b_id, service_id, float(price)))
            conn.commit()
            conn.close()

            if total_cost == 0.0:
                return False, "❌ Failed to add. Services not found in database."

            return True, f"✅ Added ({', '.join(added_list)}) to Tab! Total extra cost: ₹{total_cost:,.2f}"
        except Exception as e:
            conn.close()
            return False, f"Error: {str(e)}"

    def generate_bill(self, booking_id: int, late_checkout_hours: Optional[int] = 0) -> str:
        if booking_id is None or str(booking_id).strip() == "":
            return "⚠️ Please select a valid Booking."

        try:
            b_id = int(booking_id)
        except (ValueError, TypeError):
            return "⚠️ Booking ID must be a valid selection."

        try:
            late_h = int(late_checkout_hours) if late_checkout_hours not in [None, ""] else 0
        except (ValueError, TypeError):
            late_h = 0

        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT b.customer_id, b.room_id, b.dynamic_price, b.discount, c.name, r.room_number, r.room_type
            FROM bookings b JOIN customers c ON b.customer_id = c.customer_id JOIN rooms r ON b.room_id = r.room_id
            WHERE b.booking_id = ?
        ''', (b_id,))
        booking = cursor.fetchone()

        if not booking:
            conn.close()
            return f"❌ Booking ID {b_id} not found."

        c_id, r_id, dyn_price, discount, c_name, room_num, room_type = booking
        dyn_price = float(dyn_price)

        # Fetching Add-ons explicitly as Floats
        cursor.execute('''
            SELECT s.service_name, bs.price
            FROM booking_services bs
            JOIN addon_services s ON bs.service_id = s.service_id
            WHERE bs.booking_id = ?
        ''', (b_id,))
        services_added = cursor.fetchall()

        addon_charges = sum([float(s[1]) for s in services_added]) if services_added else 0.0
        late_penalty = (dyn_price / 24) * (late_h - 2) if late_h > 2 else 0.0
        discount_amt = float(discount) if discount else 0.0

        subtotal = dyn_price + addon_charges
        total_amount = subtotal + late_penalty - discount_amt
        total_services_used = len(services_added)

        bill = f"\n{'='*80}\n🏨 FINAL HOTEL BILL\n{'='*80}\n"
        bill += f"Customer: {c_name}\nBooking ID: {b_id} | Selected Room: {room_num} ({room_type})\n"
        bill += f"Total Services Used: {total_services_used}\n\n"
        bill += f"{'CHARGES':<40}{'AMOUNT':>15}\n{'-'*55}\n"
        bill += f"{'Room Stay Charges':<40}₹{dyn_price:>13,.2f}\n"

        if services_added:
            bill += f"\n{'--- ADD-ON SERVICES ---':<40}\n"
            for s_name, s_price in services_added:
                bill += f"  > {s_name:<36}₹{float(s_price):>13,.2f}\n"
            bill += f"{'Total Add-ons':<40}₹{addon_charges:>13,.2f}\n\n"
        else:
            bill += f"{'Add-on Services':<40}₹0.00\n"

        if late_penalty > 0: bill += f"{'Late Checkout Penalty':<40}₹{late_penalty:>13,.2f}\n"
        bill += f"{'-'*55}\n"
        bill += f"{'Subtotal':<40}₹{subtotal:>13,.2f}\n"
        if discount_amt > 0: bill += f"{'Discount':<40}-₹{discount_amt:>12,.2f}\n"
        bill += f"{'-'*55}\n"
        bill += f"{'TOTAL AMOUNT DUE':<40}₹{total_amount:>13,.2f}\n"
        bill += f"{'='*80}\n"

        # Save Bill only if it doesn't already exist for this booking to prevent duplicate entries
        cursor.execute('SELECT bill_id FROM bills WHERE booking_id = ?', (b_id,))
        if not cursor.fetchone():
            cursor.execute('INSERT INTO bills (booking_id, customer_id, room_charges, addon_charges, late_checkout_penalty, discount_amount, total_amount) VALUES (?, ?, ?, ?, ?, ?, ?)', (b_id, c_id, dyn_price, addon_charges, late_penalty, discount_amt, total_amount))
            conn.commit()

        conn.close()
        return bill

    def checkout_customer_ai(self, booking_id: int, rating: float, feedback: str) -> Tuple[bool, str]:
        if not booking_id: return False, "⚠️ Please select a valid Booking."

        try:
            b_id = int(booking_id)
        except (ValueError, TypeError):
            return False, "⚠️ Invalid Booking ID format."

        conn = self.db.get_connection()
        cursor = conn.cursor()

        sentiment_label, sentiment_score = "NEUTRAL", 0.5
        if feedback.strip():
            ai_res = self.nlp_sentiment(feedback)[0]
            sentiment_label = ai_res['label'].upper()
            if 'POSITIVE' in sentiment_label or 'LABEL_2' in sentiment_label: sentiment_label = "POSITIVE"
            elif 'NEGATIVE' in sentiment_label or 'LABEL_0' in sentiment_label: sentiment_label = "NEGATIVE"
            else: sentiment_label = "NEUTRAL"
            sentiment_score = ai_res['score']

        try:
            cursor.execute('SELECT customer_id, room_id, check_in, check_out, dynamic_price FROM bookings WHERE booking_id = ?', (b_id,))
            booking = cursor.fetchone()
            if not booking:
                conn.close()
                return False, f"❌ Booking ID {b_id} not found."

            c_id, r_id, c_in, c_out, spent = booking
            nights = max(1, (datetime.strptime(c_out, '%Y-%m-%d') - datetime.strptime(c_in, '%Y-%m-%d')).days)

            cursor.execute('INSERT INTO visit_history (customer_id, booking_id, room_id, check_in, check_out, nights, spent, rating, feedback, sentiment_label, sentiment_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (c_id, b_id, r_id, c_in, c_out, nights, spent, rating, feedback, sentiment_label, sentiment_score))
            cursor.execute("UPDATE rooms SET status = 'available' WHERE room_id = ?", (r_id,))
            cursor.execute("UPDATE bookings SET booking_status = 'completed' WHERE booking_id = ?", (b_id,))
            cursor.execute("INSERT INTO alerts (booking_id, alert_type, message) VALUES (?, ?, ?)", (b_id, "Checkout Complete", f"Guest checkout via AI processed."))
            conn.commit()
            conn.close()

            return True, f"✅ Checkout Complete!\n🧠 AI Sentiment Analysis: {sentiment_label} (Confidence: {sentiment_score*100:.1f}%)"
        except Exception as e:
            conn.close()
            return False, f"Error: {str(e)}"

    def get_analytics_dashboard(self) -> str:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT SUM(total_amount) FROM bills')
        total_rev = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT COUNT(*) FROM rooms WHERE status = 'booked'")
        booked = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM rooms")
        total_rooms = cursor.fetchone()[0]
        occ = (booked / total_rooms) * 100 if total_rooms > 0 else 0

        cursor.execute("SELECT COUNT(*) FROM booking_services")
        total_services_ordered = cursor.fetchone()[0] or 0

        cursor.execute('''
            SELECT r.room_number, COUNT(b.booking_id) as uses
            FROM rooms r JOIN bookings b ON r.room_id = b.room_id
            GROUP BY r.room_number ORDER BY uses DESC LIMIT 1
        ''')
        top_room_data = cursor.fetchone()
        top_room = top_room_data[0] if top_room_data else "N/A"

        cursor.execute("SELECT COUNT(*) FROM visit_history WHERE sentiment_label = 'POSITIVE'")
        pos = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM visit_history WHERE sentiment_label = 'NEGATIVE'")
        neg = cursor.fetchone()[0]
        conn.close()

        dash = f"\n{'='*80}\n📊 AI-POWERED ANALYTICS DASHBOARD\n{'='*80}\n\n"
        dash += f"💰 FINANCIALS: Total Revenue: ₹{total_rev:,.2f}\n"
        dash += f"🏨 OPERATIONS:\n  • Current Occupancy: {occ:.1f}%\n  • Total Services Ordered: {total_services_ordered}\n  • Most Frequently Booked Room: {top_room}\n"
        dash += f"⭐ GUEST EXPERIENCE: AI Positive Reviews: {pos} | AI Negative Reviews: {neg}\n"
        return dash

    def get_pending_alerts(self) -> str:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT a.alert_type, a.message, c.name FROM alerts a
            LEFT JOIN bookings b ON a.booking_id = b.booking_id
            LEFT JOIN customers c ON b.customer_id = c.customer_id
            ORDER BY a.created_at DESC LIMIT 10
        ''')
        alerts = cursor.fetchall()
        conn.close()

        res = f"\n{'='*80}\nSYSTEM ALERTS (Last 10)\n{'='*80}\n"
        if not alerts: return res + "No alerts."
        for a in alerts: res += f"[{a[0].upper()}] {a[1]} (Guest: {a[2] or 'System'})\n"
        return res

# ============================================================================
# GRADIO INTERFACE HELPERS
# ============================================================================

hms = HotelManagementSystem()

def get_available_room_numbers():
    conn = hms.db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT room_number, room_type FROM rooms WHERE status = 'available'")
    rooms = [f"{row[0]} ({row[1]})" for row in cursor.fetchall()]
    conn.close()
    return rooms

def get_active_booking_strings():
    conn = hms.db.get_connection()
    cursor = conn.cursor()
    # Fetch active bookings nicely formatted
    cursor.execute('''
        SELECT b.booking_id, c.name, r.room_number
        FROM bookings b
        JOIN customers c ON b.customer_id = c.customer_id
        JOIN rooms r ON b.room_id = r.room_id
        WHERE b.booking_status = 'confirmed'
    ''')
    res = [f"{row[0]} - {row[1]} (Room {row[2]})" for row in cursor.fetchall()]
    conn.close()
    return res

def extract_b_id(selection_string):
    """Safely extracts integer ID from string like '3 - John Doe (Room 101)'"""
    if not selection_string: return None
    try:
        return int(str(selection_string).split(" - ")[0])
    except:
        return None

def get_service_names():
    conn = hms.db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT service_name FROM addon_services")
    services = [row[0] for row in cursor.fetchall()]
    conn.close()
    return services

def book_room_handler(email, room_selection, cin, cout):
    room_number = room_selection.split(" ")[0] if room_selection else ""
    _, m = hms.create_booking(email, room_number, cin, cout)
    return m

def add_services_wrapper(b_selection, services_list):
    b_id = extract_b_id(b_selection)
    _, m = hms.add_services_to_booking(b_id, services_list)
    return m

def bill_wrapper(b_selection, late_h):
    b_id = extract_b_id(b_selection)
    return hms.generate_bill(b_id, late_h)

def checkout_wrapper(b_selection, rating, feedback):
    b_id = extract_b_id(b_selection)
    _, m = hms.checkout_customer_ai(b_id, rating, feedback)
    return m

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(title="Advanced Hotel Management System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏨 Enterprise Hotel Management System (AI Edition)")

    with gr.Tab("Guests"):
        with gr.Tabs():
            with gr.TabItem("Register Guest"):
                with gr.Row():
                    c_name = gr.Textbox(label="Full Name")
                    c_email = gr.Textbox(label="Email")
                with gr.Row():
                    c_phone = gr.Textbox(label="Phone")
                    c_address = gr.Textbox(label="City/Address")
                reg_btn = gr.Button("Register")
                reg_out = gr.Textbox(label="Response")
                reg_btn.click(lambda n,e,p,a: hms.add_customer(n,e,p,a)[1], inputs=[c_name, c_email, c_phone, c_address], outputs=reg_out)

            with gr.TabItem("Guest Profile Search"):
                s_email = gr.Textbox(label="Guest Email")
                s_btn = gr.Button("Search Profile")
                s_out = gr.Textbox(label="Profile & History", lines=10)
                s_btn.click(hms.get_customer_history, inputs=s_email, outputs=s_out)

    with gr.Tab("Room Inventory"):
        gr.Markdown("### 🛏️ Live Room Interactive DataFrame")
        with gr.Row():
            rt_filter = gr.Dropdown(["All", "Single", "Double", "Deluxe", "Suite", "Family", "Penthouse"], label="Filter Type", value="All")
            rf_btn = gr.Button("🔄 Refresh Inventory")
        rooms_df = gr.Dataframe(interactive=False)
        rf_btn.click(hms.get_available_rooms_df, inputs=rt_filter, outputs=rooms_df)
        rt_filter.change(hms.get_available_rooms_df, inputs=rt_filter, outputs=rooms_df)

    with gr.Tab("Bookings"):
        with gr.Tabs():
            with gr.TabItem("Master Bookings List"):
                gr.Markdown("### 📋 Complete Database of All Bookings")
                lb_btn = gr.Button("Load / Refresh Master Bookings Data")
                b_df = gr.Dataframe(interactive=False)
                lb_btn.click(hms.get_all_bookings_df, outputs=b_df)

            with gr.TabItem("Create Booking"):
                b_email = gr.Textbox(label="Guest Email")
                with gr.Row():
                    b_room = gr.Dropdown(label="Select Available Room", choices=get_available_room_numbers(), interactive=True)
                    refresh_rooms_btn = gr.Button("🔄 Refresh Room List", scale=0)
                with gr.Row():
                    b_cin = gr.Textbox(label="Check-in (YYYY-MM-DD)")
                    b_cout = gr.Textbox(label="Check-out (YYYY-MM-DD)")
                b_btn = gr.Button("Confirm Booking", variant="primary")
                b_out = gr.Textbox(label="Confirmation")

                refresh_rooms_btn.click(lambda: gr.update(choices=get_available_room_numbers()), outputs=b_room)
                b_btn.click(book_room_handler, inputs=[b_email, b_room, b_cin, b_cout], outputs=b_out)

            with gr.TabItem("AI Smart Recommendation"):
                rec_email = gr.Textbox(label="Guest Email")
                with gr.Row():
                    rec_cin = gr.Textbox(label="Check-in (YYYY-MM-DD)", placeholder="2026-08-01")
                    rec_cout = gr.Textbox(label="Check-out (YYYY-MM-DD)", placeholder="2026-08-05")
                rec_budget = gr.Number(label="Max Budget (₹)", value=20000)
                rec_btn = gr.Button("Get AI Recommendations", variant="primary")
                rec_out = gr.Textbox(label="Results", lines=10)
                rec_btn.click(hms.recommend_rooms, inputs=[rec_email, rec_cin, rec_cout, rec_budget], outputs=rec_out)

    with gr.Tab("Services"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Available Services Menu")
                v_serv_btn = gr.Button("View Services Menu")
                v_serv_out = gr.Textbox(lines=8, label="Menu List")
                v_serv_btn.click(hms.get_addon_services, outputs=v_serv_out)
            with gr.Column():
                gr.Markdown("### Add Service to Booking")

                # Replaced manual number entry with Dynamic Dropdown
                with gr.Row():
                    s_b_select = gr.Dropdown(label="Select Active Booking", choices=get_active_booking_strings(), interactive=True)
                    s_b_refresh = gr.Button("🔄 Refresh", scale=0)

                s_names = gr.CheckboxGroup(label="Select Services to Add", choices=get_service_names(), interactive=True)
                refresh_serv_btn = gr.Button("🔄 Refresh Service List")

                a_serv_btn = gr.Button("Add to Tab", variant="primary")
                a_serv_out = gr.Textbox(label="Status")

                s_b_refresh.click(lambda: gr.update(choices=get_active_booking_strings()), outputs=s_b_select)
                refresh_serv_btn.click(lambda: gr.update(choices=get_service_names()), outputs=s_names)
                a_serv_btn.click(add_services_wrapper, inputs=[s_b_select, s_names], outputs=a_serv_out)

    with gr.Tab("Billing & Checkout"):
        with gr.Tabs():
            with gr.TabItem("1. Generate Bill"):
                # Replaced manual number entry with Dynamic Dropdown
                with gr.Row():
                    bill_b_select = gr.Dropdown(label="Select Active Booking", choices=get_active_booking_strings(), interactive=True)
                    bill_b_refresh = gr.Button("🔄 Refresh", scale=0)

                late_h = gr.Number(label="Late Checkout Hours (Optional)", value=None, precision=0)
                bill_btn = gr.Button("Generate Final Bill", variant="primary")
                bill_out = gr.Textbox(lines=20)

                bill_b_refresh.click(lambda: gr.update(choices=get_active_booking_strings()), outputs=bill_b_select)
                bill_btn.click(bill_wrapper, inputs=[bill_b_select, late_h], outputs=bill_out)

            with gr.TabItem("2. AI Sentiment Checkout"):
                gr.Markdown("### Process Checkout & Run NLP on Guest Feedback")

                # Replaced manual number entry with Dynamic Dropdown
                with gr.Row():
                    co_b_select = gr.Dropdown(label="Select Active Booking to Checkout", choices=get_active_booking_strings(), interactive=True)
                    co_b_refresh = gr.Button("🔄 Refresh", scale=0)

                co_rating = gr.Slider(1, 5, step=1, label="Star Rating")
                co_feedback = gr.Textbox(label="Guest Written Feedback", lines=3, placeholder="How was your stay?")
                co_btn = gr.Button("Complete Checkout & Analyze Sentiment", variant="primary")
                co_out = gr.Textbox(label="AI System Response")

                co_b_refresh.click(lambda: gr.update(choices=get_active_booking_strings()), outputs=co_b_select)
                co_btn.click(checkout_wrapper, inputs=[co_b_select, co_rating, co_feedback], outputs=co_out)

    with gr.Tab("Analytics & Alerts"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Metrics")
                an_btn = gr.Button("Refresh Analytics", variant="primary")
                an_out = gr.Textbox(lines=10, label="Hotel Dashboard")
                an_btn.click(hms.get_analytics_dashboard, outputs=an_out)
            with gr.Column(scale=1):
                gr.Markdown("### 🔔 System Alerts")
                al_btn = gr.Button("Refresh Alerts", variant="secondary")
                al_out = gr.Textbox(lines=10, label="Recent Alerts")
                al_btn.click(hms.get_pending_alerts, outputs=al_out)

        gr.Markdown("### 📋 Master Booking List (Live Data)")
        master_df = gr.Dataframe(interactive=False)
        # When user refreshes analytics, it also refreshes the Live Master Dataframe
        an_btn.click(hms.get_all_bookings_df, outputs=master_df)

if __name__ == "__main__":
    print("🚀 Advanced Hotel Management System starting...")
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))