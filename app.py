import streamlit as st
import fitz
import logging
import boto3
import json
import os
import re
import pymysql
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import time
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms.bedrock import Bedrock
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any

from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory

identity_id = 'SESSION-ID-02'

def get_memory_from_dynamo(session_id):
  chat_history = DynamoDBChatMessageHistory(table_name="memories-dev", session_id=session_id)

  print("# message_history : ", chat_history)
  
  return ConversationBufferMemory(
    memory_key="chat_history", 
    chat_memory=chat_history, 
    return_messages=True,
    ai_prefix="AI",
    human_prefix="Human"
  ), chat_history

class StreamHandler(BaseCallbackHandler):
    def __init__(self, initial_text=""):
        self.container = st.empty()
        self.initial_text = initial_text
        self.text = initial_text

    
    def on_llm_start(self, *args: Any, **kwargs: Any):
        self.text = self.initial_text

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.text += token
        self.container.markdown(self.text, unsafe_allow_html=True)

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        st.session_state.messages.append({
            "role": "assistant",
            "type": "text",
            "content": self.text
        })

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

INIT_MESSAGE = {"role": "assistant",
                "type": "text",
                "content": """
저는 <font color='red'><b>Amazon OpenSearch</b></font> 와 <font color='red'><b>Amazon Bedrock</b></font>를 활용해서 데이터를 대신 찾아줄 <font color='red'><b>Bot</b></font> 입니다. 
<br>궁금한 것을 물어보세요.
- <font color='#32CD32;'><b>우리회사의 주문전환율에 대해 설명해줘</b></font><br>
- <font color='#32CD32;'><b>우리회사의 최근 1시간 동안 상품 별 주문전환율 top 5 데이터를 알려줘</b></font><br>
---
무엇을 도와드릴까요?"""}

select_options = ['용어가 궁금해', '데이터가 궁금해']


load_dotenv()
opensearch_username = os.getenv('OPENSEARCH_USERNAME')
opensearch_password = os.getenv('OPENSEARCH_PASSWORD')
opensearch_endpoint = os.getenv('OPENSEARCH_ENDPOINT')
index_name = os.getenv('OPENSEARCH_INDEX_NAME')
mysql_host = os.getenv('MYSQL_HOST')
mysql_port = os.getenv('MYSQL_PORT')
mysql_user = os.getenv('MYSQL_USER')
mysql_password = os.getenv('MYSQL_PASSWORD')
mysql_db = os.getenv('MYSQL_DB')

bedrock_region = 'us-west-2'
stop_record_count = 100
record_stop_yn = False
bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
bedrock_embedding_model_id = "amazon.titan-embed-text-v1"

def get_athena_client() :
    athena_client = boto3.client('athena',
                       region_name='ap-northeast-2')
    return athena_client


def get_opensearch_cluster_client():
    opensearch_client = OpenSearch(
        hosts=[{
            'host': opensearch_endpoint,
            'port': 443
        }],
        http_auth=(opensearch_username, opensearch_password),
        index_name=index_name,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30
    )
    return opensearch_client


def get_bedrock_client():
    bedrock_client = boto3.client(
        "bedrock-runtime", region_name=bedrock_region)
    return bedrock_client


def create_langchain_vector_embedding_using_bedrock(bedrock_client):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)
    return bedrock_embeddings_client


def create_opensearch_vector_search_client(bedrock_embeddings_client, _is_aoss=False):
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=bedrock_embeddings_client,
        opensearch_url=f"https://{opensearch_endpoint}",
        http_auth=(opensearch_username, opensearch_password),
        is_aoss=_is_aoss
    )
    return docsearch


def create_bedrock_llm():
    # claude-2 이하
    # bedrock_llm = Bedrock(
    #     model_id=model_version_id,
    #     client=bedrock_client,
    #     model_kwargs={'temperature': 0}
    #     )
    # bedrock_llm = BedrockChat(model_id=model_version_id, model_kwargs={'temperature': 0}, streaming=True)

    bedrock_llm = BedrockChat(
        model_id=bedrock_model_id, 
        model_kwargs={'temperature': 0},
        streaming=True,
        callbacks=[StreamHandler()]
        )
    return bedrock_llm


def get_bedrock_client():
    bedrock_client = boto3.client(
        "bedrock-runtime", region_name=bedrock_region)
    return bedrock_client


def create_vector_embedding_with_bedrock(text, bedrock_client):
    payload = {"inputText": f"{text}"}
    body = json.dumps(payload)
    modelId = "amazon.titan-embed-text-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    embedding = response_body.get("embedding")
    return {"_index": index_name, "text": text, "vector_field": embedding}

def create_opensearch_index(opensearch_client) :
    body = {
        'settings': {
            'index': {
                'number_of_shards': 3,
                'number_of_replicas': 2,
                "knn": True,
                "knn.space_type": "cosinesimil"
            }
        }
    }
    success = opensearch_client.indices.create(index_name, body=body)
    if success:
        body = {
            "properties": {
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": 1536
                },
                "text": {
                    "type": "keyword"
                }
            }
        }
        success = opensearch_client.indices.put_mapping(
            index=index_name,
            body=body
        )


def extract_sentences_from_pdf(opensearch_client, pdf_file, progress_bar, progress_text):
    try:
        logging.info(
            f"Checking if index {index_name} exists in OpenSearch cluster")

        exists = opensearch_client.indices.exists(index=index_name)

        if not exists:
            create_opensearch_index(opensearch_client)

        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        all_records = []
        for page in doc:
            all_records.append(page.get_text())

        logging.info(f"PDF LIST 개수 : {len(all_records)}")

        total_records = len(all_records)
        processed_records = 0

        bedrock_client = get_bedrock_client()

        all_json_records = []

        for record in all_records:
            if record_stop_yn and processed_records > stop_record_count:

                success, failed = bulk(opensearch_client, all_json_records)
                break

            records_with_embedding = create_vector_embedding_with_bedrock(
                record, bedrock_client)
            all_json_records.append(records_with_embedding)

            processed_records += 1
            progress = int((processed_records / total_records) * 100)
            progress_bar.progress(progress)

            if processed_records % 500 == 0 or processed_records == len(all_records):

                success, failed = bulk(opensearch_client, all_json_records)
                all_json_records = []

        progress_text.text("완료")
        logging.info("임베딩을 사용하여 레코드 생성 완료")

        return total_records
    except Exception as e:
        print(str(e))
        st.error('PDF를 임베딩 하는 과정에서 오류가 발생되었습니다.')
        return 0

def analytics_in_data(question, data):
    question = question
    bedrock_client = get_bedrock_client()
    bedrock_llm = create_bedrock_llm()

    bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(
        bedrock_client)

    opensearch_vector_search_client = create_opensearch_vector_search_client(
        bedrock_embeddings_client)
    
    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    질문을 통해 얻은 데이터이다. 
    데이터의 의미를 설명해줘. 
    데이터를 테이블 형태로 표시하고 1위 상품의 중요 숫자는 (주문전환율 등) 빨간색으로 표시해줘.
    인사이트 단어는 초록색으로 표시하고 인사이트 단어 앞에 :star2: 를 적어줘.
    데이터의 의미 부분에서 숫자는 빨간색으로 표시해줘. 색깔을 위한 태그는 span 을 사용해.
    금액은 1000자리 마다 콤마를 표시해줘.    
    과거 대화 이력의 데이터와 차이가 있다면 원인에 대해 설명해줘.
    조회수는 많지만 주문전환율이 낮은 상품에 대해 설명해줘.
    context 정보는 무시해.
    {context}
    
    # 질문
    {question}

    다음 대화와 후속 질문이 주어지면 질문에 대답해줘.
    %s

    # 데이터
    %s

    
    # Answer
    주어진 데이터는 아래와 같습니다.(테이블 형태의 데이터, 테이블 컬럼은 한글을 사용)
    ...

    데이터의 의미와 얻을 수 있는 인사이트는 다음과 같습니다. 
    - 현재 데이터
    - 과거 데이터와의 비교, 1위 순위가 변경되었다면 그 상품명을 알려줘.
    """

    data = data.replace("{","")
    data = data.replace("}","")

    memory, chat_history = get_memory_from_dynamo(identity_id)
    
    prompt_template = PromptTemplate(
            template=prompt_template % (chat_history, data), 
            input_variables=["context", "chat_history", "question"]
        )

    qa = RetrievalQA.from_chain_type(llm=bedrock_llm,
                                        chain_type="stuff",
                                        retriever=opensearch_vector_search_client.as_retriever(),
                                        # return_source_documents=True,
                                        return_source_documents=False,
                                        memory= memory,
                                        chain_type_kwargs={
                                            "prompt": prompt_template})



    response = qa(question,
                      return_only_outputs=False)
    
    return f"{response.get('result')}"


def find_answer_in_sentences(select_option, question):
    # try:
    print("# select_option : ", select_option)
    question = question
    bedrock_client = get_bedrock_client()
    bedrock_llm = create_bedrock_llm()

    bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(
        bedrock_client)

    opensearch_vector_search_client = create_opensearch_vector_search_client(
        bedrock_embeddings_client)
    
    
    prompt_template = {
        0 : """
            너는 A회사에서 근무하는 직원들에게 내부 용어, 계산식, 그리고 데이터 조회를 도와주는 챗봇이다. 
            context에 제공된 내용이 없다면 반드시 알수없다고 대답해줘.
            context에 제공된 내용이 있다면 질문한 것에 대한 정의와 계산식 그리고 데이터를 찾을 수 있는 위치를 markdown 을 사용해서 행 중심의 테이블 형식으로 대답해줘. SQL은 작성하지마.
            데이터를 찾을 수 있는 위치는 붉은색으로 표시해줘.
            {context}

            다음 대화와 후속 질문이 주어지면 질문에 대답해줘.
            %s


            * Question: {question}
            * Answer:
            """,
        1 : """
            너는 A회사에서 근무하는 직원들에게 내부 용어, 계산식, 그리고 데이터 조회를 도와주는 챗봇이다. 
            데이터 요청을 하면 context에 정확한 데이터베이스명, 테이블명, 컬럼명이 없다면 예측해서 SQL을 작성하지 말고 테이블명을 알려달라고 말해줘.
            데이터 요청을 하면 context에 정확한 테이블명과 컬럼명이 모두 있을 경우에만 AWS ATHENA에서 실행 가능한 SQL 을 MARKDOWN 코드의 SQL태그 안에 작성해줘, SQL 은 <details open><summary>SQL제목</summary>```sql\nSQL```</details> 포맷으로 작성해줘.
            {context}


            * Question: {question}
            * Answer:
            """
    }
        
    print("## prompt : ", prompt_template[select_option])
    

    prompt = PromptTemplate(
        template=prompt_template[select_option], input_variables=["context", "question"]
    )


    qa = RetrievalQA.from_chain_type(llm=bedrock_llm,
                                        chain_type="stuff",
                                        retriever=opensearch_vector_search_client.as_retriever(),
                                        # return_source_documents=True,
                                        return_source_documents=False,
                                        chain_type_kwargs={
                                            "prompt": prompt})
    
    response = qa(question,
                    return_only_outputs=False)

    return f"{response.get('result')}"

def connect_to_database():
    return pymysql.connect(
        host=mysql_host,
        port=int(mysql_port),
        user=mysql_user,
        password=mysql_password,
        database=mysql_db,
        charset='utf8mb4'
    )

def execute_query_and_return_df(sql):
    conn = connect_to_database()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=[i[0]
                              for i in cursor.description])
    finally:
        conn.close()
    return df


def execute_query_athena(client, sql):
    
    s3_client = boto3.client('s3', region_name='ap-northeast-2')
    
    s3_output = 's3://korea-summit2024/tmp/athena'
    response = client.start_query_execution(
        QueryString=sql,
        ResultConfiguration={
            'OutputLocation': s3_output,
        }
    )
    query_execution_id = response['QueryExecutionId']

    while True:
        query_status = client.get_query_execution(QueryExecutionId=query_execution_id)
        query_execution_status = query_status['QueryExecution']['Status']['State']
        
        if query_execution_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        else:
            time.sleep(2)
    
    if query_execution_status == 'SUCCEEDED':
        result_location = query_status['QueryExecution']['ResultConfiguration']['OutputLocation']
        filename = result_location.split('/')[-1]
        s3_path = f's3://{s3_output}/{filename}'
        
        df = pd.read_csv(s3_path)
        json_str = df.to_json(orient='records', lines=True)

        return json_str

    



def main():
    

    
    opensearch_client = get_opensearch_cluster_client()
    st.set_page_config(page_title='🤖 AWS Summit 2024', layout='wide')
    st.header(':blue[데이터가] _궁금해 ?!_', divider='rainbow')    

    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

    with st.sidebar:
        
        selected_option = st.sidebar.selectbox(
            '어떤 옵션을 선택하시겠습니까?',
            select_options
        )
        selected_index = select_options.index(selected_option)
        st.session_state.select_option = selected_index
        st.sidebar.markdown('---')
        st.title("🔍 RAG Embedding")
        pdf_file = st.file_uploader(
            "PDF 업로드를 통해 추가 학습을 할 수 있습니다.", type=["pdf"], key=None)
        

        if 'last_uploaded' not in st.session_state:
            st.session_state.last_uploaded = None

        if pdf_file is not None and pdf_file != st.session_state.last_uploaded:

            progress_text = st.empty()
            st.session_state['progress_bar'] = st.progress(0)
            progress_text.text("RAG(OpenSearch) 임베딩 중...")
            record_cnt = extract_sentences_from_pdf(
                opensearch_client, pdf_file, st.session_state['progress_bar'], progress_text)
            if record_cnt > 0:
                st.session_state['processed'] = True
                st.session_state['record_cnt'] = record_cnt
                st.session_state['progress_bar'].progress(100)
                st.session_state.last_uploaded = pdf_file
                st.success(f"{record_cnt} Vector 임베딩 완료!")
        
        if st.button("기존 인덱스 삭제"):
            exists = opensearch_client.indices.exists(index=index_name)

            if exists:
                opensearch_client.indices.delete(index=index_name)
                create_opensearch_index(opensearch_client)            
                logging.info("OpenSearch index successfully deleted")
                st.success("OpenSearch 인덱스가 성공적으로 삭제/생성되었습니다.")
            else :
                create_opensearch_index(opensearch_client)            
                st.success("OpenSearch 인덱스가 성공적으로 생성되었습니다.")

        st.sidebar.markdown('---')

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [INIT_MESSAGE]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"], unsafe_allow_html=True)

    question = st.chat_input("Say something")

    if question:
        st.session_state.messages.append({"role": "user",
                                          "type": "text",
                                          "content": question})
        with st.chat_message("user"):
            st.markdown(question, unsafe_allow_html=True)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = find_answer_in_sentences(st.session_state.select_option, question)

            with st.spinner("데이터 조회 중..."):

                try :
                    sql_queries = re.findall(r'```sql(.*?)```', answer, re.DOTALL)

                    if len(sql_queries) > 0:
                        sql = sql_queries[0]
                        df = execute_query_athena(get_athena_client(), sql)
                        analytics_in_data(question, df)
                except Exception as e:                    
                    print("# 에러", str(e))
                    pass


if __name__ == "__main__":
    main()
