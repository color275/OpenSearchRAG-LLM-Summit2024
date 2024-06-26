# cd /home/ec2-user/OpenSearchRAG-LLM-Summit2024/gen_data
# nohup python generate.py &

import random
import datetime
import pymysql
import time
import os
from faker import Faker
from dotenv import load_dotenv

# Example usage
# 1. product
# 2. basket
# 3. order
weights = [7, 2, 1]
product_ids = list(range(1, 21))  # 상품 ID 1부터 20까지
product_weights = [9, 1, 3, 2, 3, 4, 3, 2, 1, 2, 3, 4, 2, 4, 3, 2, 1, 2, 3, 4]

fake = Faker()

load_dotenv()
db_config = {
    'host': os.getenv('MYSQL_HOST'),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_DB'),
}

def connect_to_database(config):
    try:
        connection = pymysql.connect(**config)
        return connection
    except pymysql.MySQLError as err:
        print(f"Error: {err}")
        return None

def get_max_order_id(connection):
    query = "SELECT MAX(order_id) AS max_id FROM orders"
    with connection.cursor() as cursor:
        try:
            cursor.execute(query)
            result = cursor.fetchone()
            if result[0] is None:
                return 1
            return result[0] + 1
        except pymysql.MySQLError as err:
            print(f"Error: {err}")
            return 1

def insert_order_to_database(connection, order_id, promo_id, order_cnt, order_price, order_dt, customer_id, product_id):
    query = """
    INSERT INTO orders (promo_id, order_cnt, order_price, order_dt, customer_id, product_id)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (promo_id, order_cnt, order_price, order_dt, customer_id, product_id)
    
    with connection.cursor() as cursor:
        try:
            cursor.execute(query, values)
            connection.commit()
            print("Order inserted successfully.")
        except pymysql.MySQLError as err:
            print(f"Error: {err}")

def generate_log_entry(order_id_counter, timestamp, fake):
    client_ip = fake.ipv4()
    product_id = random.choices(product_ids, weights=product_weights, k=1)[0]  # 가중치에 따라 상품 ID 선택
    customer_id = random.randint(1, 100)
    request_types = ['products', 'basket', 'order']
    request_type = random.choices(request_types, weights=weights, k=1)[0]

    if request_type == 'products':
        request_line = f'"GET /products?product_id={product_id}&customer_id={customer_id} HTTP/1.1"'
    elif request_type == 'basket':
        request_line = f'"GET /basket?product_id={product_id}&customer_id={customer_id} HTTP/1.1"'
    else:  # 'order'
        request_line = f'"GET /order?order_id={order_id_counter}&product_id={product_id}&customer_id={customer_id} HTTP/1.1"'
        order_id_counter += 1

    log_entry = f"{timestamp} {client_ip} - - {request_line} 200 1576\n"
    return log_entry, order_id_counter, request_type, product_id, customer_id

def write_logs_with_db_insertion(db_config, cnt_per_sec):
    db_connection = connect_to_database(db_config)
    if db_connection is None:
        return
    
    order_id_counter = get_max_order_id(db_connection)
    
    today_date = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"accesslog/access_log_{today_date}.log"

    i = 0
    while True:
        timestamp = datetime.datetime.now().strftime("|%Y-%m-%d %H:%M:%S|")
        log_entry, order_id_counter, request_type, product_id, customer_id = generate_log_entry(order_id_counter, timestamp, fake)
        
        with open(filename, 'a') as file:
            file.write(log_entry)
            
        if request_type == "order":
            promo_id = f'PROMO{random.randint(1, 20):02d}'
            order_cnt = random.randint(1, 10)
            order_price = random.randint(5, 50) * 1000
            order_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_order_to_database(db_connection, order_id_counter, promo_id, order_cnt, order_price, order_dt, customer_id, product_id)
        
        i += 1
        if i % cnt_per_sec == 0:
            time.sleep(1)
            i = 0



# 초당 5건
cnt_per_sec = 5

write_logs_with_db_insertion(db_config, cnt_per_sec)
