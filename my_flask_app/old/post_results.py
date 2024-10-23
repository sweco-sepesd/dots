import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError

FME_TOKEN = '27875d5873837122939a850985fe3f624f7ab093'
FME_SERVER_WEB_URL = 'http://localhost'

def get_json(url):
    request = Request(url, headers={'Authorization': f'fmetoken token={FME_TOKEN}'})
    response = urlopen(request)
    assert 200 == response.status and 'application/json;charset=UTF-8' == response.headers.get('Content-Type')
    return json.load(response)

def upload_resource(data, filename, resource, path, create_directories=False, overwrite=False):
    query = ''
    if create_directories or overwrite:
        query = f'?createDirectories={str(create_directories).lower()}&overwrite={str(overwrite).lower()}'
    url = f'{FME_SERVER_WEB_URL}/fmerest/v3/resources/connections/{resource}/filesys/{path}{query}'
    headers={
          'Authorization': f'fmetoken token={FME_TOKEN}'
        , 'Content-Disposition': f'attachment;filename="{filename}"'
        , 'Content-Type': 'application/octet-stream'
    }
    request = Request(url, data=data, headers=headers)
    response = urlopen(request)
    assert 201 == response.status and 'application/json;charset=UTF-8' == response.headers.get('Content-Type')
    return json.load(response)
    
    
    
def submit_job(repository, workspace, parameters):
    url = f'{FME_SERVER_WEB_URL}/fmerest/v3/transformations/submit/{repository}/{workspace}'
    headers={'Authorization': f'fmetoken token={FME_TOKEN}', 'Content-Type': 'application/json'}
    body = {'NMDirectives': {}, 'TMDirectives': {}, 'publishedParameters': [{'name': k, 'value': v} for k,v in parameters.items()]}
    data = json.dumps(body, ensure_ascii=False).encode('utf8')
    request = Request(url, data=data, headers=headers)
    response = urlopen(request)
    assert 202 == response.status and 'application/json;charset=UTF-8' == response.headers.get('Content-Type')
    return json.load(response)

import os.path
fp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'samples', 'peter.segerstedt@sweco.se.png'))
with open(fp, 'rb') as fin:
    data = fin.read()
    try:
        path = 'highscore/results'
        filename = 'peter.segerstedt@sweco.se.png'
        file_info = upload_resource(data, filename, 'FME_SHAREDRESOURCE_DATA', path, overwrite=True)
        print(file_info)
        # NOTE: Dataset parameters for file based reader formats are often expected to be an array (that's how multiple files are handled)
        job_info = submit_job('CONNECT_THE_DOTS', 'evaluate_contribution.fmw', {'SourceDataset_PNGRASTER': [f'$(FME_SHAREDRESOURCE_DATA){path}/{filename}']})
        print(job_info)
    except HTTPError as e:
        print(e.msg, e.name, e.reason, e.status)
