import bcrypt
import json
import os
from datetime import datetime

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Load existing users
users = {}
if os.path.exists('data/users.json'):
    with open('data/users.json', 'r') as f:
        users = json.load(f)

# Create admin user
admin_id = "1"
users[admin_id] = {
    "username": "admin",
    "email": "admin@jobcheck.com",
    "password_hash": hash_password("admin123"),
    "is_admin": True,
    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Save to file
os.makedirs('data', exist_ok=True)
with open('data/users.json', 'w') as f:
    json.dump(users, f, indent=2)

print("✅ Admin user created successfully!")
print("Username: admin")
print("Password: admin123")