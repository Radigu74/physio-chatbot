import os
import csv

def save_user_data(name, email, phone, country):
    # Construct the file path within the mounted volume
    file_path = os.path.join('/data', 'user_logs.csv')
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'email', 'phone', 'country']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'name': name,
            'email': email,
            'phone': phone,
            'country': country
        })
