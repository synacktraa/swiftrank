import json
import requests
from pathlib import Path

ENDPOINT = "http://127.0.0.1:12345/rerank"
BODY = {
  "model": "ms-marco-TinyBERT-L-2-v2",
  "contexts": {},
  "query": "string",
  "threshold": None,
  "schema": {
    "pre": None,
    "ctx": None,
    "post": None
  }
}
FINAL_OUTPUT = [
    "Jujutsu Kaisen 2nd Season",
    "Jujutsu Kaisen 2nd Season Recaps",
    "Jujutsu Kaisen",
    "Jujutsu Kaisen Official PV",
    "Jujutsu Kaisen 0 Movie",
    "Shingeki no Kyojin Season 2",
    "Shingeki no Kyojin Season 3 Part 2",
    "Shingeki no Kyojin Season 3",
    "Kimi ni Todoke 2nd Season",
    "Shingeki no Kyojin: The Final Season"
]

files_path = Path(__file__).parent.parent / 'files'

def read_file_as_context_field(name: str, rl: bool = False):
    content = (files_path / name).read_text()
    if rl:
        return content.split('\n')
    return json.loads(content)
     

def test_http_exception_empty_array_or_object():
    response = requests.post(url=ENDPOINT, json=BODY)
    assert response.status_code == 422
    assert response.json()['detail'] == "contexts field cannot be an empty array or object"

def test_http_exception_model_not_available():
    response = requests.post(
        url=ENDPOINT, json=BODY | {'model': 'no-model', 'contexts': ["non", "empty"]}
    )
    assert response.status_code == 404
    assert response.json()['detail'] == "'no-model' model is not available"

def test_http_exception_empty_array_after_pre_processing():
    response = requests.post(
        url=ENDPOINT, json=BODY | {'contexts': {
            "categories": [
                {
                    "type": "anime",
                    "items": []
                }
            ]
        }, 'schema': {'pre': '.categories[].items'}}
    )
    assert response.status_code == 422
    assert response.json()['detail'] == "Empty array after pre-processing"

def test_http_exception_pre_processing_must_result_into_array():
    response = requests.post(
        url=ENDPOINT, json=BODY | {'contexts': {
            "categories": [
                {
                    "type": "anime",
                    "items": ['non', 'empty']
                }
            ]
        }, 'schema': {'pre': '.categories[].type'}}
    )
    assert response.status_code == 422
    assert response.json()['detail'] == "Pre-processing must result into an array of objects"
   
def test_http_exception_expected_arrary_of_string_or_object():
    response = requests.post(
        url=ENDPOINT, json=BODY | {'contexts': {
            "categories": [
                {
                    "type": "anime",
                    "items": ['non', 'empty']
                }
            ]
        }}
    )
    assert response.status_code == 422
    assert response.json()['detail'] == "Expected an array of string or object. 'pre' schema might help"

def test_arrary_as_input():
    response = requests.post(
        url=ENDPOINT, json=BODY | {
            'query': "Jujutsu Season 2",
            'contexts': read_file_as_context_field('contexts', rl=True)}
    )

    assert response.status_code == 200
    assert response.json() == FINAL_OUTPUT

def test_object_as_input():
    response = requests.post(
        url=ENDPOINT, json=BODY | {
            'query': "Jujutsu Season 2",
            'contexts': read_file_as_context_field('contexts.json'),
            'threshold': 0.9,
            'schema': {
                'pre': '.categories[].items',
                'ctx': '.name',
                'post': '.name'
            }
        }
    )
    assert response.status_code == 200
    assert response.json() == FINAL_OUTPUT[0:3]
