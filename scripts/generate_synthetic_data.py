
import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

try:
    from faker import Faker
except ImportError:
    raise SystemExit("Run `pip install Faker` first (see requirements.txt).")

fake = Faker("en_IN")

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

STORE_TYPES = {
    "kirana": {
        "name_templates": ["{name} General Store", "{name} Kirana Store", "New {name} Stores"],
        "items": [
            ("Milk", "दूध", "dudh", (25, 70)),
            ("Rice", "चावल", "chawal", (60, 220)),
            ("Wheat Flour", "आटा", "aata", (40, 150)),
            ("Sugar", "चीनी", "cheeni", (40, 60)),
            ("Tea Powder", "चाय पत्ती", "chai patti", (60, 200)),
            ("Biscuits", "बिस्कुट", "biscuit", (10, 60)),
            ("Cooking Oil", "तेल", "tel", (120, 250)),
            ("Salt", "नमक", "namak", (10, 25)),
            ("Soap", "साबुन", "sabun", (20, 60)),
            ("Detergent", "डिटर्जेंट", "detergent", (50, 180)),
            ("Onion", "प्याज", "pyaz", (20, 60)),
            ("Potato", "आलू", "aalu", (15, 40)),
            ("Eggs (dozen)", "अंडे", "ande", (60, 90)),
            ("Bread", "ब्रेड", "bread", (30, 50)),
        ],
    },
    "restaurant": {
        "name_templates": ["{name} Restaurant", "Hotel {name}", "{name} Dhaba", "{name} Cafe"],
        "items": [
            ("Paneer Butter Masala", None, None, (180, 320)),
            ("Butter Naan", None, None, (30, 60)),
            ("Veg Biryani", None, None, (150, 280)),
            ("Masala Dosa", None, None, (80, 150)),
            ("Chicken Tikka", None, None, (220, 400)),
            ("Dal Makhani", None, None, (150, 250)),
            ("Cold Coffee", None, None, (60, 120)),
            ("Gulab Jamun", None, None, (60, 100)),
            ("Roti", None, None, (12, 25)),
            ("Lassi", None, None, (50, 90)),
        ],
    },
    "medical": {
        "name_templates": ["{name} Medical Store", "{name} Pharmacy", "{name} Chemist"],
        "items": [
            ("Paracetamol 500mg (strip)", None, None, (15, 40)),
            ("Cough Syrup", None, None, (60, 150)),
            ("Antiseptic Liquid", None, None, (50, 120)),
            ("Bandage Roll", None, None, (20, 50)),
            ("Vitamin C Tablets", None, None, (80, 200)),
            ("Hand Sanitizer", None, None, (60, 150)),
            ("Face Mask (pack of 5)", None, None, (30, 80)),
            ("ORS Sachet", None, None, (15, 30)),
        ],
    },
    "supermarket": {
        "name_templates": ["{name} Supermarket", "{name} Mart", "Big {name} Bazaar"],
        "items": [
            ("Basmati Rice 1kg", None, None, (90, 250)),
            ("Toor Dal 1kg", None, None, (110, 180)),
            ("Shampoo 200ml", None, None, (90, 250)),
            ("Toothpaste", None, None, (40, 120)),
            ("Instant Noodles (pack)", None, None, (12, 30)),
            ("Frozen Peas 500g", None, None, (60, 100)),
            ("Ketchup Bottle", None, None, (80, 150)),
            ("Chocolate Bar", None, None, (20, 80)),
            ("Cornflakes", None, None, (120, 250)),
            ("Curd 400g", None, None, (30, 55)),
        ],
    },
    "fuel": {
        "name_templates": ["{name} Petrol Pump", "{name} Fuel Station", "Indian Oil - {name}"],
        "items": [
            ("Petrol (litres)", None, None, (95, 105)),  # price PER LITRE, handled specially
        ],
    },
    "clothing": {
        "name_templates": ["{name} Garments", "{name} Fashion Store", "{name} Textiles"],
        "items": [
            ("Cotton Shirt", None, None, (400, 1200)),
            ("Jeans", None, None, (700, 2000)),
            ("Kurta", None, None, (500, 1500)),
            ("T-Shirt", None, None, (250, 800)),
            ("Saree", None, None, (800, 4000)),
            ("Socks (pair)", None, None, (50, 150)),
        ],
    },
}

INDIAN_CITIES_STATES = [
    ("Bhopal", "Madhya Pradesh"), ("Indore", "Madhya Pradesh"), ("Jaipur", "Rajasthan"),
    ("Lucknow", "Uttar Pradesh"), ("Patna", "Bihar"), ("Pune", "Maharashtra"),
    ("Nagpur", "Maharashtra"), ("Ahmedabad", "Gujarat"), ("Surat", "Gujarat"),
    ("Chandigarh", "Punjab"), ("Ranchi", "Jharkhand"), ("Bhubaneswar", "Odisha"),
    ("Guwahati", "Assam"), ("Coimbatore", "Tamil Nadu"), ("Kochi", "Kerala"),
    ("Nashik", "Maharashtra"), ("Varanasi", "Uttar Pradesh"), ("Amritsar", "Punjab"),
]

PAYMENT_METHODS = ["UPI", "Cash", "Card", "PhonePe", "Google Pay", "Paytm", "BharatPe"]

GST_RATES = [0, 5, 12, 18, 28]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rand_date():
    d = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 600))
    return d


def format_date_variant(d):
    fmts = ["%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d.%m.%Y"]
    return d.strftime(random.choice(fmts))


def money(v):
    return round(v, 2)


def maybe_rupee_symbol():
    # Simulates inconsistent currency formatting across receipts/OCR
    return random.choice(["₹", "Rs.", "Rs", "INR", ""])


def ocr_noise(text: str, rate: float = 0.04) -> str:
    """Injects light OCR-like character confusion."""
    subs = {
        "0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5",
        "₹": "Rs", "8": "B", "B": "8", "2": "Z",
    }
    out = []
    for ch in text:
        if ch in subs and random.random() < rate:
            out.append(subs[ch])
        else:
            out.append(ch)
    return "".join(out)


def pick_language_style():
    return random.choices(["english", "hindi", "hinglish"], weights=[0.6, 0.15, 0.25])[0]


def item_label(item_tuple, lang):
    en, hi, hing, _ = item_tuple
    if lang == "hindi" and hi:
        return hi
    if lang == "hinglish" and hing:
        return hing.title()
    return en


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_one(idx: int, add_noise: bool = True):
    store_type = random.choice(list(STORE_TYPES.keys()))
    catalog = STORE_TYPES[store_type]
    vendor = random.choice(catalog["name_templates"]).format(name=fake.last_name())
    city, state = random.choice(INDIAN_CITIES_STATES)
    lang = pick_language_style()
    date = rand_date()
    payment = random.choice(PAYMENT_METHODS)
    currency_sym = maybe_rupee_symbol()

    n_items = random.randint(1, 6)
    chosen = random.sample(catalog["items"], k=min(n_items, len(catalog["items"])))

    line_items = []
    subtotal = 0.0
    for it in chosen:
        name_en, hi, hing, price_range = it
        label = item_label(it, lang)
        if store_type == "fuel":
            litres = round(random.uniform(2, 30), 2)
            price_per_l = round(random.uniform(*price_range), 2)
            line_total = money(litres * price_per_l)
            line_items.append({
                "name": "Petrol",
                "qty": litres,
                "unit": "litres",
                "unit_price": price_per_l,
                "price": line_total,
            })
            display_label = f"{label} @{price_per_l}/L x {litres}L"
        else:
            qty = random.choice([1, 1, 1, 2, 3])
            unit_price = round(random.uniform(*price_range), 2)
            line_total = money(unit_price * qty)
            line_items.append({
                "name": name_en,
                "qty": qty,
                "unit_price": unit_price,
                "price": line_total,
            })
            display_label = label if qty == 1 else f"{label} x{qty}"
        subtotal += line_total

    subtotal = money(subtotal)
    use_gst_split = random.random() < 0.5
    gst_rate = random.choice(GST_RATES)
    gst_amount = money(subtotal * gst_rate / 100)
    cgst = money(gst_amount / 2)
    sgst = money(gst_amount - cgst)
    total = money(subtotal + gst_amount)

    # ---- Build the raw receipt text (what OCR would produce) ----
    lines = [vendor.upper(), f"{city}, {state}", ""]
    for li, disp in zip(line_items, [None] * len(line_items)):
        pass
    for orig, li in zip(chosen, line_items):
        label = item_label(orig, lang)
        qty = li.get("qty", 1)
        price_str = f"{currency_sym}{li['price']}"
        if qty and qty != 1 and "unit" not in li:
            lines.append(f"{label} x{qty}".ljust(28) + price_str)
        else:
            lines.append(label.ljust(28) + price_str)

    lines.append("")
    lines.append(f"Subtotal".ljust(28) + f"{currency_sym}{subtotal}")
    if gst_rate > 0:
        if use_gst_split:
            lines.append(f"CGST".ljust(28) + f"{currency_sym}{cgst}")
            lines.append(f"SGST".ljust(28) + f"{currency_sym}{sgst}")
        else:
            lines.append(f"GST ({gst_rate}%)".ljust(28) + f"{currency_sym}{gst_amount}")
    lines.append(f"Total".ljust(28) + f"{currency_sym}{total}")
    lines.append(f"{payment}".ljust(28) + f"{currency_sym}{total}")
    lines.append("")
    lines.append(f"Date: {format_date_variant(date)}")

    receipt_text = "\n".join(lines)
    if add_noise:
        receipt_text = ocr_noise(receipt_text)

    # ---- Ground-truth structured label (clean, canonical) ----
    label = {
        "vendor": vendor,
        "location": f"{city}, {state}",
        "store_type": store_type,
        "items": [
            {"name": li["name"], "qty": li.get("qty", 1), "unit_price": li["unit_price"], "price": li["price"]}
            for li in line_items
        ],
        "subtotal": subtotal,
        "gst": gst_amount,
        "cgst": cgst if (gst_rate > 0 and use_gst_split) else 0,
        "sgst": sgst if (gst_rate > 0 and use_gst_split) else 0,
        "total": total,
        "payment_method": payment,
        "date": date.strftime("%Y-%m-%d"),
        "currency": "INR",
    }

    instruction = (
        "Extract the following fields from this Indian receipt as JSON: "
        "vendor, location, items (name, qty, unit_price, price), subtotal, gst, "
        "cgst, sgst, total, payment_method, date (YYYY-MM-DD), currency.\n\n"
        f"Receipt:\n{receipt_text}"
    )

    return {"receipt_text": receipt_text, "label": label, "instruction": instruction}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="total number of receipts to generate")
    ap.add_argument("--out", type=str, default="data/synthetic", help="output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-noise", action="store_true", help="disable OCR noise injection")
    args = ap.parse_args()

    random.seed(args.seed)
    Faker.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [generate_one(i, add_noise=not args.no_noise) for i in range(args.n)]
    random.shuffle(records)

    n_train = int(args.n * 0.85)
    n_val = int(args.n * 0.075)
    train, val, test = records[:n_train], records[n_train:n_train + n_val], records[n_train + n_val:]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split)} records to {path}")

    print(f"\nDone. Total: {len(records)} (train={len(train)}, val={len(val)}, test={len(test)})")


if __name__ == "__main__":
    main()
