# OpenSearchRAG-LLM-Summit2024

### env
```
cd OpenSearchRAG-LLM-Summit2024
vi .env
OPENSEARCH_USERNAME=
OPENSEARCH_PASSWORD=
OPENSEARCH_ENDPOINT=
OPENSEARCH_INDEX_NAME=
MYSQL_HOST=
MYSQL_PORT=
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DB=
```

### Data Gen Setup
```bash
# python env
cd OpenSearchRAG-LLM-Summit2024/gen_data
python3.10 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

# mysql create table
mysql -u[userename] -p -h [RDS_WRITER_ENDPOINT]
create database [database_name]
use [database_name]
source ecommerce_backup.sql
exit

# accesslog setup
sudo mkdir -p /var/log/accesslog/
sudo chown ec2-user:ec2-user /var/log/accesslog/

# data generate
nohup python generate.py &
```

### Streamlit Setup
```
deactivate
cd OpenSearchRAG-LLM-Summit2024
python3.10 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start
```
export AWS_DEFAULT_REGION='us-west-2'
streamlit run app.py
```


### Capture
![](img/2024-04-06-15-09-36.png)
![](img/2024-04-06-15-08-49.png)